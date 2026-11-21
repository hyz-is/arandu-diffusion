//go:build sdcpp

package diffusion

/*
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "stable-diffusion.h"

// A package with //export directives may only declare, not define, in its
// preamble, so the trampolines below are static: each translation unit gets its
// own copy and none of them is an exported symbol.
extern void goDiffusionProgress(uintptr_t handle, char* stage, char* detail, int64_t completed, int64_t total);
extern void goDiffusionLog(int level, char* text);

static void diffusion_progress(const sd_generation_progress_t* p, void* data) {
	goDiffusionProgress((uintptr_t)data, (char*)p->stage, (char*)p->detail, p->completed, p->total);
}

static void diffusion_log(enum sd_log_level_t level, const char* text, void* data) {
	(void)data;
	goDiffusionLog((int)level, (char*)text);
}

static void diffusion_register_progress(sd_ctx_t* ctx, uintptr_t handle) {
	sd_set_generation_progress_callback(ctx, diffusion_progress, (void*)handle);
}

static void diffusion_clear_progress(sd_ctx_t* ctx) {
	sd_set_generation_progress_callback(ctx, NULL, NULL);
}

// The library draws a text progress bar on stderr unless a legacy progress
// callback is registered. Progress reaches the application through the
// context-local callback instead, so this one only keeps the bar off.
static void diffusion_quiet(int step, int steps, float time, void* data) {
	(void)step; (void)steps; (void)time; (void)data;
}

static void diffusion_register_log(void) {
	sd_set_log_callback(diffusion_log, NULL);
	sd_set_progress_callback(diffusion_quiet, NULL);
}

static void diffusion_lora_at(sd_lora_t* arr, size_t i, const char* path, float multiplier) {
	arr[i].is_high_noise = false;
	arr[i].multiplier = multiplier;
	arr[i].path = path;
}

static void diffusion_tensor_at(const sd_tensor_t** arr, size_t i, const sd_tensor_t* t) {
	arr[i] = t;
}
*/
import "C"

import (
	"context"
	"fmt"
	"runtime"
	"runtime/cgo"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// Linked reports whether this build links stable-diffusion.cpp.
const Linked = true

func init() {
	// The library logs through one process-wide callback. It is routed to Go
	// once, and what is done with the lines is decided by whoever called
	// WithLogger last.
	C.diffusion_register_log()
}

// logs keeps the last warnings and errors of the library, which are the text of
// a failed run, and forwards every line to the logger the application gave.
var logs struct {
	mu     sync.Mutex
	recent []string
	sink   func(level, text string)
}

var logLevels = [...]string{"debug", "verbose", "info", "warn", "error"}

//export goDiffusionLog
func goDiffusionLog(level C.int, text *C.char) {
	line := strings.TrimRight(C.GoString(text), "\r\n")
	name := "info"
	if int(level) >= 0 && int(level) < len(logLevels) {
		name = logLevels[level]
	}
	logs.mu.Lock()
	sink := logs.sink
	if name == "warn" || name == "error" {
		if len(line) > 240 {
			line = line[:240]
		}
		logs.recent = append(logs.recent, line)
		if len(logs.recent) > 8 {
			logs.recent = logs.recent[len(logs.recent)-8:]
		}
	}
	logs.mu.Unlock()
	if sink != nil {
		sink(name, line)
	}
}

// recentLogs returns the kept lines with the prompt struck out: a library
// message that quotes what the person wrote is not a message to put in an error.
func recentLogs(prompt string) string {
	logs.mu.Lock()
	lines := append([]string(nil), logs.recent...)
	logs.mu.Unlock()
	for i, l := range lines {
		if prompt != "" {
			l = strings.ReplaceAll(l, prompt, "[prompt]")
		}
		lines[i] = l
	}
	return strings.Join(lines, "; ")
}

func clearLogs() {
	logs.mu.Lock()
	logs.recent = nil
	logs.mu.Unlock()
}

// sdEngine is the real Engine: one stable-diffusion.cpp context.
type sdEngine struct {
	// sem is held for the length of a generation. A channel and not a mutex so a
	// caller waiting for its turn can give up when its context does.
	sem    chan struct{}
	ctx    *C.sd_ctx_t
	handle cgo.Handle
	loras  []LoRA
	info   Info

	closed    atomic.Bool
	closeOnce sync.Once
	target    atomic.Pointer[progressTarget]
}

// progressTarget is where the library's reports go for the run in flight.
type progressTarget struct {
	fn    func(Progress)
	start time.Time
}

//export goDiffusionProgress
func goDiffusionProgress(handle C.uintptr_t, stage, detail *C.char, completed, total C.int64_t) {
	e, ok := cgo.Handle(handle).Value().(*sdEngine)
	if !ok {
		return
	}
	t := e.target.Load()
	if t == nil || t.fn == nil {
		return
	}
	// A panic must not unwind through the C frames below this call.
	defer func() { _ = recover() }()
	t.fn(Progress{
		Stage:     C.GoString(stage),
		Detail:    C.GoString(detail),
		Completed: int64(completed),
		Total:     int64(total),
		Elapsed:   time.Since(t.start),
	})
}

// Load reads the model set into memory and returns the Engine over it.
//
// It validates the set, digests every file, and creates the library context,
// which maps the weights. That is the expensive call: seconds for a small model,
// longer for a large one. Do it once and keep the Engine.
func Load(set ModelSet, opts ...Option) (Engine, error) {
	if err := set.Validate(); err != nil {
		return nil, err
	}
	o, err := newLoadOptions(opts)
	if err != nil {
		return nil, err
	}
	if o.logger != nil {
		logs.mu.Lock()
		logs.sink = o.logger
		logs.mu.Unlock()
	}
	started := time.Now()
	files, err := describeAll(set, !o.noDigests)
	if err != nil {
		return nil, err
	}

	// The library keeps its last diagnostic per thread, so the thread that makes
	// a call has to be the one that reads the diagnostic.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	clearLogs()

	opt := C.sd_ctx_opt_create()
	if opt == nil {
		return nil, fmt.Errorf("%w: out of memory creating the options", ErrGeneration)
	}
	defer C.sd_opt_free(opt)
	cfg := &optionSet{p: opt}
	if set.Checkpoint != "" {
		cfg.str("model_path", set.Checkpoint)
	}
	cfg.str("diffusion_model_path", set.Diffusion)
	cfg.str("vae_path", set.VAE)
	cfg.str("clip_l_path", set.CLIPL)
	cfg.str("clip_g_path", set.CLIPG)
	cfg.str("t5xxl_path", set.T5XXL)
	cfg.str("llm_path", set.LLM)
	cfg.str("backend", o.backend)
	if o.threads > 0 {
		cfg.integer("n_threads", int64(o.threads))
	}
	if o.flashAttn {
		cfg.boolean("diffusion_flash_attn", true)
	}
	if cfg.err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidModelSet, cfg.err)
	}

	var ctx *C.sd_ctx_t
	if st := C.sd_ctx_create(opt, &ctx); st != C.SD_STATUS_OK {
		detail := lastError()
		if lines := recentLogs(""); lines != "" {
			detail += " (" + lines + ")"
		}
		if st == C.SD_STATUS_OUT_OF_MEMORY {
			return nil, fmt.Errorf("%w: loading the model ran out of memory: %s", ErrGeneration, detail)
		}
		return nil, fmt.Errorf("%w: the library could not load the set: %s", ErrInvalidModelSet, detail)
	}
	if !bool(C.sd_ctx_supports_image_generation(ctx)) {
		C.sd_ctx_free(ctx)
		return nil, fmt.Errorf("%w: the model does not generate images", ErrInvalidModelSet)
	}

	name := set.Name
	if name == "" && len(files) > 0 {
		for _, f := range files {
			if f.Role == "checkpoint" || f.Role == "diffusion" {
				name = f.Name
				break
			}
		}
	}
	threads := o.threads
	if threads == 0 {
		threads = int(C.sd_get_num_physical_cores())
	}
	e := &sdEngine{
		sem:   make(chan struct{}, 1),
		ctx:   ctx,
		loras: append([]LoRA(nil), set.LoRAs...),
		info: Info{
			Name:         name,
			Backend:      linkedBackend,
			Threads:      threads,
			Library:      C.GoString(C.sd_version()) + " " + C.GoString(C.sd_commit()),
			ModelVersion: C.GoString(C.sd_get_model_version_name(ctx)),
			Models:       files,
			LoadDuration: time.Since(started),
		},
	}
	e.handle = cgo.NewHandle(e)
	C.diffusion_register_progress(ctx, C.uintptr_t(e.handle))
	return e, nil
}

// Info describes what is loaded.
func (e *sdEngine) Info() Info { return e.info }

// Close cancels a run in flight, waits for it to stop and frees the model.
func (e *sdEngine) Close() error {
	e.closeOnce.Do(func() {
		e.closed.Store(true)
		// The context is freed only below, after the semaphore is ours, so this
		// call cannot reach a freed pointer.
		C.sd_cancel_generation(e.ctx, C.enum_sd_cancel_mode_t(C.SD_CANCEL_ALL))
		e.sem <- struct{}{}
		C.diffusion_clear_progress(e.ctx)
		C.sd_ctx_free(e.ctx)
		e.ctx = nil
		e.handle.Delete()
	})
	return nil
}

// Generate runs one request.
func (e *sdEngine) Generate(ctx context.Context, req Request) (*Result, error) {
	if e.closed.Load() {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// All the work that can fail for the caller's reasons happens before the
	// semaphore is taken and before a byte of the model is touched.
	p, err := prepare(req, randomSeed)
	if err != nil {
		return nil, err
	}

	select {
	case e.sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-e.sem }()
	if e.closed.Load() {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	clearLogs()

	opt := C.sd_img_gen_opt_create()
	if opt == nil {
		return nil, fmt.Errorf("%w: out of memory creating the options", ErrGeneration)
	}
	defer C.sd_opt_free(opt)
	if err := e.configure(opt, p); err != nil {
		return nil, err
	}

	started := time.Now()
	e.target.Store(&progressTarget{fn: req.OnProgress, start: started})
	defer e.target.Store(nil)

	// The watcher turns the caller's cancellation into the library's. It is
	// joined before the cancellation state is reset, so a late cancel cannot
	// land on the next run.
	done := make(chan struct{})
	var watcher sync.WaitGroup
	watcher.Add(1)
	go func() {
		defer watcher.Done()
		select {
		case <-ctx.Done():
			C.sd_cancel_generation(e.ctx, C.enum_sd_cancel_mode_t(C.SD_CANCEL_ALL))
		case <-done:
		}
	}()
	var result *C.sd_result_t
	status := C.sd_generate_image(e.ctx, opt, &result)
	close(done)
	watcher.Wait()
	C.sd_cancel_generation(e.ctx, C.enum_sd_cancel_mode_t(C.SD_CANCEL_RESET))
	sampling := time.Since(started)
	detail := lastError()
	if result != nil {
		defer C.sd_result_free(result)
	}

	if status != C.SD_STATUS_OK {
		return nil, e.generationError(ctx, status, detail, req.Prompt)
	}
	if C.sd_result_get_image_count(result) < 1 {
		return nil, fmt.Errorf("%w: the library returned no image", ErrGeneration)
	}
	generated, err := readImage(C.sd_result_get_image(result, 0))
	if err != nil {
		return nil, err
	}
	if generated.w != p.w || generated.h != p.h {
		return nil, fmt.Errorf("%w: the library returned %dx%d for a request of %dx%d", ErrGeneration, generated.w, generated.h, p.w, p.h)
	}
	png, err := finish(p, generated)
	if err != nil {
		return nil, err
	}
	return &Result{
		PNG: png, Width: p.w, Height: p.h, Seed: p.seed, Mode: p.mode,
		Metrics: Metrics{
			Total: time.Since(started), Sampling: sampling, Steps: p.steps,
			PeakRSSBytes: peakRSS(), Backend: e.info.Backend, Threads: e.info.Threads,
			ModelVersion: e.info.ModelVersion, Models: e.info.Models,
		},
	}, nil
}

// configure writes the request into the library's options.
func (e *sdEngine) configure(opt *C.sd_opt_t, p *prepared) error {
	req := p.req
	o := &optionSet{p: opt}
	o.str("prompt", req.Prompt)
	o.str("negative_prompt", req.NegativePrompt)
	o.integer("width", int64(p.w))
	o.integer("height", int64(p.h))
	o.integer("seed", p.seed)
	o.integer("sample.steps", int64(p.steps))
	if req.CFG > 0 {
		o.float("sample.guidance.txt_cfg", req.CFG)
	}
	if req.Guidance > 0 {
		o.float("sample.guidance.distilled_guidance", req.Guidance)
	}
	o.str("sample.method", req.Sampler)
	o.str("sample.scheduler", req.Scheduler)
	if req.ClipSkip > 0 {
		o.integer("clip_skip", int64(req.ClipSkip))
	}
	if p.init != nil {
		o.tensor("init_image", *p.init)
		if req.Strength > 0 {
			o.float("strength", req.Strength)
		}
	}
	if p.mask != nil {
		o.tensor("mask_image", *p.mask)
	}
	if len(p.refs) > 0 {
		o.tensors("ref_images", p.refs)
	}
	if len(e.loras) > 0 {
		o.loras("loras", e.loras)
	}
	if o.err != nil {
		return o.err
	}
	return nil
}

// generationError turns a failed status into an error a caller can match.
func (e *sdEngine) generationError(ctx context.Context, status C.sd_status_t, detail, prompt string) error {
	if status == C.SD_STATUS_CANCELLED {
		if err := ctx.Err(); err != nil {
			return err
		}
		return ErrClosed
	}
	text := detail
	if prompt != "" {
		text = strings.ReplaceAll(text, prompt, "[prompt]")
	}
	if lines := recentLogs(prompt); lines != "" {
		text += " (" + lines + ")"
	}
	switch status {
	case C.SD_STATUS_INVALID_ARGUMENT, C.SD_STATUS_UNKNOWN_KEY, C.SD_STATUS_TYPE_MISMATCH, C.SD_STATUS_OUT_OF_RANGE:
		return fmt.Errorf("%w: %s", ErrInvalidRequest, text)
	case C.SD_STATUS_UNSUPPORTED:
		return fmt.Errorf("%w: %s", ErrUnsupported, text)
	default:
		return fmt.Errorf("%w: status %d: %s", ErrGeneration, int(status), text)
	}
}

// lastError reads the library's diagnostic for the calling thread.
func lastError() string {
	if msg := C.sd_get_last_error(); msg != nil {
		return C.GoString(msg)
	}
	return ""
}

// readImage copies a result tensor into Go memory as RGB.
func readImage(t *C.sd_tensor_t) (pixels, error) {
	if t == nil {
		return pixels{}, fmt.Errorf("%w: the result has no image", ErrGeneration)
	}
	if C.sd_tensor_type(t) != C.SD_TENSOR_U8 || C.sd_tensor_rank(t) != 3 {
		return pixels{}, fmt.Errorf("%w: the image is not an 8-bit [C,W,H] tensor", ErrGeneration)
	}
	c, w, h := int(C.sd_tensor_dim(t, 0)), int(C.sd_tensor_dim(t, 1)), int(C.sd_tensor_dim(t, 2))
	size := int(C.sd_tensor_byte_size(t))
	if c < 1 || c > 4 || w <= 0 || h <= 0 || size != c*w*h {
		return pixels{}, fmt.Errorf("%w: the image tensor is inconsistent (%dx%dx%d, %d bytes)", ErrGeneration, c, w, h, size)
	}
	raw := C.GoBytes(C.sd_tensor_data(t), C.int(size))
	if c == 3 {
		return pixels{w: w, h: h, channels: 3, data: raw}, nil
	}
	out := pixels{w: w, h: h, channels: 3, data: make([]byte, w*h*3)}
	for i := 0; i < w*h; i++ {
		switch c {
		case 1, 2:
			out.data[i*3], out.data[i*3+1], out.data[i*3+2] = raw[i*c], raw[i*c], raw[i*c]
		default:
			copy(out.data[i*3:i*3+3], raw[i*c:i*c+3])
		}
	}
	return out, nil
}

// optionSet writes keys into one library options object and keeps the first
// error, so a run of setters reads as a list and is checked once.
type optionSet struct {
	p   *C.sd_opt_t
	err error
}

func (o *optionSet) fail(key string, st C.sd_status_t) {
	if st == C.SD_STATUS_OK || o.err != nil {
		return
	}
	switch st {
	case C.SD_STATUS_UNSUPPORTED:
		o.err = fmt.Errorf("%w: option %s: %s", ErrUnsupported, key, lastError())
	default:
		o.err = fmt.Errorf("%w: option %s: %s", ErrInvalidRequest, key, lastError())
	}
}

// str sets a string key. An empty value leaves the library's default, which is
// not the same as an empty string: the library refuses some of those.
func (o *optionSet) str(key, value string) {
	if o.err != nil || value == "" {
		return
	}
	k, v := C.CString(key), C.CString(value)
	defer C.free(unsafe.Pointer(k))
	defer C.free(unsafe.Pointer(v))
	o.fail(key, C.sd_opt_set_string(o.p, k, v))
}

func (o *optionSet) integer(key string, value int64) {
	if o.err != nil {
		return
	}
	k := C.CString(key)
	defer C.free(unsafe.Pointer(k))
	o.fail(key, C.sd_opt_set_int(o.p, k, C.int64_t(value)))
}

func (o *optionSet) float(key string, value float64) {
	if o.err != nil {
		return
	}
	k := C.CString(key)
	defer C.free(unsafe.Pointer(k))
	o.fail(key, C.sd_opt_set_float(o.p, k, C.double(value)))
}

func (o *optionSet) boolean(key string, value bool) {
	if o.err != nil {
		return
	}
	k := C.CString(key)
	defer C.free(unsafe.Pointer(k))
	o.fail(key, C.sd_opt_set_bool(o.p, k, C.bool(value)))
}

// newTensor copies pixels into a library tensor, [channels, width, height].
func newTensor(px pixels) (*C.sd_tensor_t, error) {
	if len(px.data) == 0 || len(px.data) != px.channels*px.w*px.h {
		return nil, fmt.Errorf("%w: an image has no pixels", ErrInvalidRequest)
	}
	shape := [3]C.int64_t{C.int64_t(px.channels), C.int64_t(px.w), C.int64_t(px.h)}
	t := C.sd_tensor_create(C.sd_tensor_type_t(C.SD_TENSOR_U8), 3, &shape[0], unsafe.Pointer(&px.data[0]), C.size_t(len(px.data)))
	if t == nil {
		return nil, fmt.Errorf("%w: creating a tensor: %s", ErrGeneration, lastError())
	}
	return t, nil
}

func (o *optionSet) tensor(key string, px pixels) {
	if o.err != nil {
		return
	}
	t, err := newTensor(px)
	if err != nil {
		o.err = err
		return
	}
	// The options retain the tensor after a successful setter, so this reference
	// is released either way.
	defer C.sd_tensor_free(t)
	k := C.CString(key)
	defer C.free(unsafe.Pointer(k))
	o.fail(key, C.sd_opt_set_tensor(o.p, k, t))
}

func (o *optionSet) tensors(key string, pxs []pixels) {
	if o.err != nil || len(pxs) == 0 {
		return
	}
	arr := (**C.sd_tensor_t)(C.malloc(C.size_t(len(pxs)) * C.size_t(unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(arr))
	made := make([]*C.sd_tensor_t, 0, len(pxs))
	defer func() {
		for _, t := range made {
			C.sd_tensor_free(t)
		}
	}()
	for i, px := range pxs {
		t, err := newTensor(px)
		if err != nil {
			o.err = err
			return
		}
		made = append(made, t)
		C.diffusion_tensor_at((**C.sd_tensor_t)(unsafe.Pointer(arr)), C.size_t(i), t)
	}
	k := C.CString(key)
	defer C.free(unsafe.Pointer(k))
	o.fail(key, C.sd_opt_set_tensors(o.p, k, (**C.sd_tensor_t)(unsafe.Pointer(arr)), C.size_t(len(pxs))))
}

func (o *optionSet) loras(key string, ls []LoRA) {
	if o.err != nil || len(ls) == 0 {
		return
	}
	arr := (*C.sd_lora_t)(C.malloc(C.size_t(len(ls)) * C.size_t(unsafe.Sizeof(C.sd_lora_t{}))))
	defer C.free(unsafe.Pointer(arr))
	paths := make([]*C.char, 0, len(ls))
	defer func() {
		for _, p := range paths {
			C.free(unsafe.Pointer(p))
		}
	}()
	for i, l := range ls {
		path := C.CString(l.Path)
		paths = append(paths, path)
		C.diffusion_lora_at(arr, C.size_t(i), path, C.float(l.scale()))
	}
	k := C.CString(key)
	defer C.free(unsafe.Pointer(k))
	o.fail(key, C.sd_opt_set_loras(o.p, k, arr, C.size_t(len(ls))))
}

var _ Engine = (*sdEngine)(nil)
