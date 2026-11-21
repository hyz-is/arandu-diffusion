package diffusion

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"sync"
	"time"
)

// Fake is an Engine that needs no model and no C toolchain. It is what an
// application's tests pass where production passes the engine Load returns.
//
//	fake := diffusion.NewFake()
//	svc := services.NewThumbnails(db, fake)
//	...
//	fake.Requests()[0].Prompt // what the service asked for
//
// It honours the whole contract of Engine except the picture: it validates the
// request like the real engine, decodes the same images, reports progress, stops
// on a cancelled context, keeps the masked area of an inpaint identical to the
// original, and answers PNG bytes of the size asked for. The pixels it paints
// are a pattern derived from the seed and the prompt, the same on every run.
type Fake struct {
	mu       sync.Mutex
	script   []Answer
	requests []Request
	closed   bool
	info     Info

	// Delay is how long each sampling step takes, so a test can cancel a run that
	// is still going. Zero is no delay.
	Delay time.Duration
}

// Answer is one scripted answer. A Fake spends its answers in order and, once
// they run out, generates.
type Answer struct {
	// Err, when set, is returned instead of an image.
	Err error
}

var _ Engine = (*Fake)(nil)

// NewFake builds a fake. The answers are given first, one per call, and the
// calls after them generate.
func NewFake(answers ...Answer) *Fake {
	return &Fake{
		script: answers,
		info: Info{
			Name:    "fake",
			Backend: "fake",
			Threads: 1,
			Models:  []ModelFile{{Role: "checkpoint", Name: "fake.gguf", Bytes: 1, SHA256: fakeDigest, Quantization: "Q8_0"}},
		},
	}
}

// fakeDigest is the digest of the one model file a Fake claims to have.
var fakeDigest = func() string {
	sum := sha256.Sum256([]byte("diffusion fake model"))
	return hex.EncodeToString(sum[:])
}()

// Push appends answers to the script.
func (f *Fake) Push(answers ...Answer) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.script = append(f.script, answers...)
}

// Requests are the requests received, in order, including the ones refused.
func (f *Fake) Requests() []Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]Request(nil), f.requests...)
}

// Info describes the fake model.
func (f *Fake) Info() Info {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.info
}

// Close marks the fake closed; later calls answer ErrClosed.
func (f *Fake) Close() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.closed = true
	return nil
}

// Generate records the request and gives the next scripted answer or, with none
// left, a generated picture.
func (f *Fake) Generate(ctx context.Context, req Request) (*Result, error) {
	f.mu.Lock()
	f.requests = append(f.requests, req)
	closed := f.closed
	var answer *Answer
	if len(f.script) > 0 {
		a := f.script[0]
		f.script = f.script[1:]
		answer = &a
	}
	info := f.info
	f.mu.Unlock()

	if closed {
		return nil, ErrClosed
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	p, err := prepare(req, randomSeed)
	if err != nil {
		return nil, err
	}
	if answer != nil && answer.Err != nil {
		return nil, answer.Err
	}

	started := time.Now()
	for step := 1; step <= p.steps; step++ {
		if f.Delay > 0 {
			select {
			case <-time.After(f.Delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if req.OnProgress != nil {
			req.OnProgress(Progress{Stage: "sampling", Completed: int64(step), Total: int64(p.steps), Elapsed: time.Since(started)})
		}
	}
	sampling := time.Since(started)

	out, err := finish(p, pattern(p.w, p.h, p.seed, req.Prompt))
	if err != nil {
		return nil, err
	}
	return &Result{
		PNG: out, Width: p.w, Height: p.h, Seed: p.seed, Mode: p.mode,
		Metrics: Metrics{
			Total: time.Since(started), Sampling: sampling, Steps: p.steps,
			PeakRSSBytes: peakRSS(), Backend: info.Backend, Threads: info.Threads,
			ModelVersion: "fake", Models: info.Models,
		},
	}, nil
}

// finish turns what the library generated into what the caller receives: for an
// inpaint, the generated picture with the original put back under the mask.
func finish(p *prepared, generated pixels) ([]byte, error) {
	if p.mode == ModeInpaint {
		generated = composite(*p.init, generated, *p.mask)
	}
	return encodePNG(generated)
}

// pattern paints a picture that depends only on the size, the seed and the
// prompt. It is a diagonal gradient with a hash-derived palette.
func pattern(w, h int, seed int64, prompt string) pixels {
	sum := sha256.Sum256(append(binary.LittleEndian.AppendUint64(nil, uint64(seed)), prompt...))
	out := pixels{w: w, h: h, channels: 3, data: make([]byte, w*h*3)}
	for y := range h {
		for x := range w {
			t := (x*255/w + y*255/h) / 2
			i := (y*w + x) * 3
			out.data[i] = byte((t + int(sum[0])) % 256)
			out.data[i+1] = byte((t*2 + int(sum[1])) % 256)
			out.data[i+2] = byte((255 - t + int(sum[2])) % 256)
		}
	}
	return out
}

// randomSeed draws the seed of a request that asked for a random one. The
// result is non-negative and below 2^62, so it survives a round trip through a
// JSON number and a signed database column.
func randomSeed() (int64, error) {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return 0, errors.New("diffusion: no randomness for a seed: " + err.Error())
	}
	return int64(binary.LittleEndian.Uint64(b[:]) >> 2), nil
}
