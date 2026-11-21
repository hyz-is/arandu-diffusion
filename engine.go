package diffusion

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"time"
)

// Engine is what generates images: one loaded model set, one generation at a
// time.
//
// Two implementations ship. Load returns the real one, which links
// stable-diffusion.cpp through cgo and runs the model inside the calling
// process; it exists only when the application builds with the `sdcpp` tag, and
// without the tag Load answers ErrBackendUnavailable. Fake is the scripted one,
// for an application's own tests, and needs no model and no C toolchain.
//
// An Engine is safe for concurrent use. Generations are serialised, because the
// library holds one context per model set and a second caller would race it:
// the second Generate waits its turn and gives up when its own context does.
type Engine interface {
	// Generate runs one request and returns the PNG with the metrics of the run.
	// A cancelled context stops the run at the next step and answers ctx.Err().
	Generate(ctx context.Context, req Request) (*Result, error)

	// Info describes what is loaded. It never touches the model.
	Info() Info

	// Close gives the model back. A closed Engine answers ErrClosed.
	Close() error
}

// Sentinel errors. Match them with errors.Is: each is wrapped with the detail
// of the call that produced it.
var (
	// ErrInvalidRequest is a request the engine refuses before spending any
	// compute: a missing prompt, a size that is not a multiple of eight, a mask
	// without the image it belongs to.
	ErrInvalidRequest = errors.New("diffusion: invalid request")

	// ErrInvalidModelSet is a model set that cannot be loaded: no model, both
	// kinds of model, a file that is missing or is not a regular file.
	ErrInvalidModelSet = errors.New("diffusion: invalid model set")

	// ErrBackendUnavailable is Load called in a build without the `sdcpp` tag.
	ErrBackendUnavailable = errors.New("diffusion: this build does not link stable-diffusion.cpp; build with -tags sdcpp (see the README)")

	// ErrUnsupported is a request the loaded model cannot serve, for example
	// reference images given to a model that has no editing mode.
	ErrUnsupported = errors.New("diffusion: the loaded model does not support this request")

	// ErrClosed is a call on an Engine that was closed.
	ErrClosed = errors.New("diffusion: the engine is closed")

	// ErrGeneration is a failure inside the library after the request was
	// accepted: out of memory, a corrupt weight, a graph that did not run. The
	// library's own diagnostic is part of the message.
	ErrGeneration = errors.New("diffusion: generation failed")
)

// Mode is what a request asks for, derived from the inputs it carries.
type Mode string

const (
	// ModeTxt2Img generates from the prompt alone.
	ModeTxt2Img Mode = "txt2img"
	// ModeImg2Img starts from Request.Init and keeps what Strength allows.
	ModeImg2Img Mode = "img2img"
	// ModeInpaint regenerates only where Request.Mask is white; where it is
	// black the pixels of Request.Init come back identical.
	ModeInpaint Mode = "inpaint"
	// ModeEdit conditions the generation on Request.References, as FLUX.2 klein
	// and Kontext do.
	ModeEdit Mode = "edit"
)

// LoRA is a low-rank adapter applied on top of the base weights.
type LoRA struct {
	// Path is the adapter file, .safetensors or .gguf.
	Path string
	// Scale is the multiplier; zero means 1.
	Scale float64
}

// ModelSet names the files one model is made of. Exactly one of Checkpoint and
// Diffusion is set; the rest depend on the family.
//
//	SD 1.x, 2.x, SDXL   Checkpoint (it carries the encoders and the VAE)
//	FLUX.1              Diffusion + VAE + CLIPL + T5XXL
//	FLUX.2 klein        Diffusion + VAE + LLM (Qwen3)
//	Z-Image, Qwen-Image Diffusion + VAE + LLM
//
// Every path is read by the engine and by nothing else: the set is configuration
// the application owns, never something a request names.
type ModelSet struct {
	// Name labels the model in every run. Empty means the base name of the
	// primary file.
	Name string

	// Checkpoint is a single-file model: weights, text encoders and VAE together.
	Checkpoint string
	// Diffusion is the diffusion model alone, GGUF-quantised or not.
	Diffusion string

	// VAE is the autoencoder, when the model file does not carry one.
	VAE string
	// CLIPL, CLIPG and T5XXL are the text encoders of the families that use them.
	CLIPL, CLIPG, T5XXL string
	// LLM is the language-model text encoder of the families built on one.
	LLM string

	// LoRAs are applied to every run.
	LoRAs []LoRA
}

// primary is the file the model is named and digested by.
func (s ModelSet) primary() string {
	if s.Checkpoint != "" {
		return s.Checkpoint
	}
	return s.Diffusion
}

// files lists every file of the set with the role it plays, in a stable order.
func (s ModelSet) files() []modelRef {
	refs := []modelRef{
		{"checkpoint", s.Checkpoint},
		{"diffusion", s.Diffusion},
		{"vae", s.VAE},
		{"clip_l", s.CLIPL},
		{"clip_g", s.CLIPG},
		{"t5xxl", s.T5XXL},
		{"llm", s.LLM},
	}
	for _, l := range s.LoRAs {
		refs = append(refs, modelRef{"lora", l.Path})
	}
	out := refs[:0]
	for _, r := range refs {
		if r.path != "" {
			out = append(out, r)
		}
	}
	return out
}

type modelRef struct{ role, path string }

// Validate refuses a set that cannot be loaded, naming the field. It touches the
// file system only to stat each path.
func (s ModelSet) Validate() error {
	if (s.Checkpoint == "") == (s.Diffusion == "") {
		return fmt.Errorf("%w: exactly one of Checkpoint and Diffusion is required", ErrInvalidModelSet)
	}
	for _, l := range s.LoRAs {
		if l.Path == "" {
			return fmt.Errorf("%w: a LoRA has no path", ErrInvalidModelSet)
		}
		if l.Scale < 0 || l.Scale > 10 {
			return fmt.Errorf("%w: LoRA scale %v is outside 0..10", ErrInvalidModelSet, l.Scale)
		}
	}
	for _, ref := range s.files() {
		info, err := os.Stat(ref.path)
		if err != nil {
			return fmt.Errorf("%w: %s: %v", ErrInvalidModelSet, ref.role, err)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("%w: %s %q is not a regular file", ErrInvalidModelSet, ref.role, ref.path)
		}
	}
	return nil
}

// loraScale is the multiplier with its default applied.
func (l LoRA) scale() float64 {
	if l.Scale == 0 {
		return 1
	}
	return l.Scale
}

// Request is one generation. The zero value of every optional field means the
// library's default for the loaded model.
type Request struct {
	// Prompt is required, at most MaxPromptBytes.
	Prompt string
	// NegativePrompt is what to steer away from. Models that take no guidance
	// ignore it.
	NegativePrompt string

	// Width and Height are in pixels, multiples of 8, between MinSide and
	// MaxSide. Zero means the size of Init when there is one, else 512.
	Width, Height int

	// Seed makes the run repeatable. Zero is a valid seed; a negative one asks
	// for a random seed, which Result.Seed reports.
	Seed int64
	// Steps is the sampling step count; zero is the library default (20).
	Steps int
	// CFG is the text guidance scale; zero is the library default (7). Distilled
	// models such as FLUX.2 klein want 1.
	CFG float64
	// Guidance is the distilled-guidance scale of the FLUX.1 family; zero is the
	// library default.
	Guidance float64
	// Sampler and Scheduler are the library's names ("euler", "dpm++2m",
	// "karras"). Empty selects the best default for the loaded model. The library
	// refuses a name it does not know, and the refusal is ErrInvalidRequest.
	Sampler, Scheduler string
	// ClipSkip is the number of final text-encoder layers to skip; zero is the
	// library default.
	ClipSkip int

	// Init is the starting image for img2img and the image to repaint for
	// inpainting: PNG or JPEG bytes. Its sides must be multiples of 8.
	Init []byte
	// Strength is how much img2img may change Init, in (0, 1]; zero is 0.75.
	Strength float64
	// Mask is PNG or JPEG bytes of the same size as Init. White is repainted and
	// black is kept; grey blends. Transparent pixels are read as black, so a mask
	// with an alpha channel keeps what it does not cover.
	Mask []byte

	// References are the images an editing model conditions on, at most
	// MaxReferences, PNG or JPEG bytes.
	References [][]byte

	// OnProgress is called from the generating goroutine as the library reports
	// its stages. It must return quickly and must not call the Engine.
	OnProgress func(Progress)
}

// The limits a request is held to. They bound what one call can ask of the
// machine; they are not a statement about what a model can do.
const (
	// MaxPromptBytes bounds Prompt and NegativePrompt.
	MaxPromptBytes = 8 << 10
	// MinSide and MaxSide bound Width and Height.
	MinSide = 64
	MaxSide = 4096
	// MaxSteps bounds Steps.
	MaxSteps = 150
	// MaxReferences bounds References.
	MaxReferences = 4
	// MaxImageBytes bounds each of Init, Mask and every reference, encoded.
	MaxImageBytes = 32 << 20
	// MaxImagePixels bounds each decoded image.
	MaxImagePixels = 4096 * 4096
)

// nameRe is what a sampler or scheduler name may look like. It is a shape check:
// the library holds the list of names, and a list kept here would drift from it.
var nameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9_.+-]{0,31}$`)

// Mode reports what the request asks for.
func (r Request) Mode() Mode {
	switch {
	case len(r.Init) > 0 && len(r.Mask) > 0:
		return ModeInpaint
	case len(r.References) > 0:
		return ModeEdit
	case len(r.Init) > 0:
		return ModeImg2Img
	default:
		return ModeTxt2Img
	}
}

// Progress is one report from the library.
type Progress struct {
	// Stage is "preparing", "conditioning", "sampling", "vae_encode" or
	// "vae_decode".
	Stage string
	// Detail narrows the stage, for example "positive_prompt". It may be empty.
	Detail string
	// Completed and Total count units of the stage; for "sampling" they are
	// steps. Total is zero when unknown.
	Completed, Total int64
	// Elapsed is the time since the run started.
	Elapsed time.Duration
}

// Result is one finished generation.
type Result struct {
	// PNG is the image, encoded.
	PNG []byte
	// Width and Height are the pixel size of PNG.
	Width, Height int
	// Seed is the seed the run used, the one to pass to repeat it.
	Seed int64
	// Mode is what the run did.
	Mode Mode
	// Metrics is what the run cost and what produced it.
	Metrics Metrics
}

// Metrics is the record of one run, the part of a Result that is kept.
type Metrics struct {
	// Total is the wall time of Generate, decoding and encoding included.
	Total time.Duration
	// Sampling is the wall time inside the library.
	Sampling time.Duration
	// Steps is the sampling step count that was requested, after the default.
	Steps int
	// PeakRSSBytes is the high-water mark of the process's resident memory when
	// the run finished. It covers the whole process since it started, the load of
	// the model included, so it bounds the run from above and does not isolate
	// it. Zero where the platform has no way to read it.
	PeakRSSBytes uint64
	// Backend is the compute backend linked: "cpu", "metal", "cuda" or "vulkan".
	Backend string
	// Threads is the CPU thread count the engine was loaded with.
	Threads int
	// ModelVersion is the family the library recognised, such as "FLUX.2 klein".
	ModelVersion string
	// Models are the files that produced the image, with their digests.
	Models []ModelFile
}

// ModelFile identifies one file of the model set.
type ModelFile struct {
	// Role is "checkpoint", "diffusion", "vae", "clip_l", "clip_g", "t5xxl",
	// "llm" or "lora".
	Role string
	// Name is the base name of the file; the directory is never recorded.
	Name string
	// Bytes is the file size.
	Bytes int64
	// SHA256 is the hex digest of the whole file; empty when the engine was
	// loaded WithoutDigests.
	SHA256 string
	// Quantization is the dominant weight type of a GGUF file, such as "Q8_0" or
	// "F16"; empty for other formats.
	Quantization string
}

// Info describes a loaded engine.
type Info struct {
	// Name is ModelSet.Name, or the base name of the primary file.
	Name string
	// Backend and Threads are as in Metrics.
	Backend string
	Threads int
	// Library is the stable-diffusion.cpp version string; empty for the Fake.
	Library string
	// ModelVersion is as in Metrics.
	ModelVersion string
	// Models are the files of the set.
	Models []ModelFile
	// LoadDuration is how long Load took, digests included.
	LoadDuration time.Duration
}

// Option configures Load.
type Option func(*loadOptions)

type loadOptions struct {
	threads   int
	backend   string
	flashAttn bool
	noDigests bool
	logger    func(level, text string)
}

// WithThreads sets the CPU thread count. Zero, the default, lets the library
// pick the number of physical cores.
func WithThreads(n int) Option { return func(o *loadOptions) { o.threads = n } }

// WithBackend names the device assignment the library accepts for its backend
// option, for example a device from sd_list_devices. Empty is the library's own
// choice.
func WithBackend(spec string) Option { return func(o *loadOptions) { o.backend = spec } }

// WithFlashAttention turns on flash attention in the diffusion model, which
// saves memory and time on the models that support it.
func WithFlashAttention() Option { return func(o *loadOptions) { o.flashAttn = true } }

// WithoutDigests skips hashing the model files at load. The digests are what
// make a run attributable to the exact weights, so leave them on except in a
// test that loads the same multi-gigabyte file over and over.
func WithoutDigests() Option { return func(o *loadOptions) { o.noDigests = true } }

// WithLogger receives the library's log lines. Level is "debug", "verbose",
// "info", "warn" or "error". Without it nothing is printed; warnings and errors
// are still kept for the message of a failed run.
func WithLogger(fn func(level, text string)) Option { return func(o *loadOptions) { o.logger = fn } }

func newLoadOptions(opts []Option) (loadOptions, error) {
	var o loadOptions
	for _, opt := range opts {
		opt(&o)
	}
	if o.threads < 0 || o.threads > 1024 {
		return o, fmt.Errorf("%w: thread count %d", ErrInvalidModelSet, o.threads)
	}
	return o, nil
}

// Validate refuses a request that cannot be run, before any compute is spent.
// Generate calls it, and an application can call it earlier, at its own edge.
func (r Request) Validate() error {
	bad := func(format string, a ...any) error {
		return fmt.Errorf("%w: %s", ErrInvalidRequest, fmt.Sprintf(format, a...))
	}
	if r.Prompt == "" {
		return bad("the prompt is empty")
	}
	if len(r.Prompt) > MaxPromptBytes || len(r.NegativePrompt) > MaxPromptBytes {
		return bad("a prompt is longer than %d bytes", MaxPromptBytes)
	}
	if r.Width < 0 || r.Height < 0 {
		return bad("the size is negative")
	}
	if (r.Width != 0 && (r.Width < MinSide || r.Width > MaxSide || r.Width%8 != 0)) ||
		(r.Height != 0 && (r.Height < MinSide || r.Height > MaxSide || r.Height%8 != 0)) {
		return bad("width and height must be multiples of 8 between %d and %d", MinSide, MaxSide)
	}
	if r.Steps < 0 || r.Steps > MaxSteps {
		return bad("steps must be between 0 and %d", MaxSteps)
	}
	if r.CFG < 0 || r.CFG > 30 || r.Guidance < 0 || r.Guidance > 30 {
		return bad("guidance scales must be between 0 and 30")
	}
	if r.ClipSkip < 0 || r.ClipSkip > 12 {
		return bad("clip skip must be between 0 and 12")
	}
	if r.Strength < 0 || r.Strength > 1 {
		return bad("strength must be in (0, 1]")
	}
	if r.Sampler != "" && !nameRe.MatchString(r.Sampler) {
		return bad("the sampler name %q is not a name", r.Sampler)
	}
	if r.Scheduler != "" && !nameRe.MatchString(r.Scheduler) {
		return bad("the scheduler name %q is not a name", r.Scheduler)
	}
	if len(r.Mask) > 0 && len(r.Init) == 0 {
		return bad("a mask needs the image it belongs to")
	}
	if len(r.References) > MaxReferences {
		return bad("at most %d reference images", MaxReferences)
	}
	for _, b := range append([][]byte{r.Init, r.Mask}, r.References...) {
		if len(b) > MaxImageBytes {
			return bad("an image is larger than %d bytes", MaxImageBytes)
		}
	}
	return nil
}
