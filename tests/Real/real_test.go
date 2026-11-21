//go:build sdcpp

// Package real_test runs the engine against a real model. It needs the `sdcpp`
// build tag and the archives from `make archives`, and each test skips itself,
// saying why, when the model file it needs is not there:
//
//	DIFFUSION_MODEL   a single-file checkpoint, for example
//	                  ~/.cache/hyzis-dev/ferflow/models/sd15-inpaint-Q8_0.gguf
//	DIFFUSION_THREADS CPU threads; default is the library's
//
//	DIFFUSION_FLUX2_DIFFUSION, DIFFUSION_FLUX2_VAE, DIFFUSION_FLUX2_LLM
//	                  a FLUX.2 klein set, for the reference-image edit
//
//	CGO_ENABLED=1 go test -tags sdcpp -count=1 -v ./tests/Real/
package real_test

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	diffusion "github.com/hyz-is/arandu-diffusion"
)

func threads() int {
	n, _ := strconv.Atoi(os.Getenv("DIFFUSION_THREADS"))
	return n
}

// model skips the test unless DIFFUSION_MODEL names a file.
func model(t *testing.T) string {
	t.Helper()
	path := os.Getenv("DIFFUSION_MODEL")
	if path == "" {
		t.Skip("DIFFUSION_MODEL is not set: no model to run")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("DIFFUSION_MODEL %s is absent: %v", path, err)
	}
	return path
}

// engine loads the model once per test binary: loading maps gigabytes, and the
// tests below only differ in what they ask of it.
var (
	once    sync.Once
	shared  diffusion.Engine
	loadErr error
)

func engine(t *testing.T) diffusion.Engine {
	t.Helper()
	path := model(t)
	once.Do(func() {
		shared, loadErr = diffusion.Load(diffusion.ModelSet{Checkpoint: path}, diffusion.WithThreads(threads()))
	})
	if loadErr != nil {
		t.Fatalf("loading %s: %v", path, loadErr)
	}
	return shared
}

func TestMain(m *testing.M) {
	code := m.Run()
	if shared != nil {
		_ = shared.Close()
	}
	os.Exit(code)
}

func decode(t *testing.T, b []byte) *image.NRGBA {
	t.Helper()
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("the result is not a png: %v", err)
	}
	// The standard library decodes an opaque 8-bit PNG as *image.RGBA; compare in
	// one representation.
	n := image.NewNRGBA(img.Bounds())
	draw.Draw(n, n.Bounds(), img, img.Bounds().Min, draw.Src)
	return n
}

// gradient is a picture the tests can start from without paying for a
// generation: every pixel differs from its neighbours, so a pixel that moved
// cannot hide.
func gradient(t *testing.T, w, h int) []byte {
	t.Helper()
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			src.SetNRGBA(x, y, color.NRGBA{R: uint8(x * 255 / w), G: uint8(y * 255 / h), B: uint8((x*7 + y*13) % 256), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestTxt2ImgOnCPU(t *testing.T) {
	e := engine(t)
	info := e.Info()
	t.Logf("model %s (%s) backend=%s threads=%d load=%s library=%s", info.Name, info.ModelVersion, info.Backend, info.Threads, info.LoadDuration, info.Library)
	if info.Backend != "cpu" {
		t.Fatalf("backend = %q, the CPU build is what this test is for", info.Backend)
	}

	var mu sync.Mutex
	var last diffusion.Progress
	steps := 0
	res, err := e.Generate(context.Background(), diffusion.Request{
		Prompt: "a red apple on a wooden table, studio photo", NegativePrompt: "blurry, text",
		Width: 256, Height: 256, Seed: 42, Steps: 4, CFG: 7, Sampler: "euler",
		OnProgress: func(p diffusion.Progress) {
			mu.Lock()
			defer mu.Unlock()
			last = p
			if p.Stage == "sampling" {
				steps++
			}
		},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	img := decode(t, res.PNG)
	if img.Bounds().Dx() != 256 || img.Bounds().Dy() != 256 {
		t.Fatalf("size = %v, want 256x256", img.Bounds())
	}
	if res.Seed != 42 || res.Mode != diffusion.ModeTxt2Img {
		t.Fatalf("seed=%d mode=%s", res.Seed, res.Mode)
	}
	// A picture of nothing is a failure that still returns bytes: all one colour.
	distinct := map[color.NRGBA]bool{}
	for y := 0; y < 256; y += 4 {
		for x := 0; x < 256; x += 4 {
			distinct[img.NRGBAAt(x, y)] = true
		}
	}
	if len(distinct) < 200 {
		t.Fatalf("only %d distinct colours in a sample of the picture: it is blank", len(distinct))
	}
	m := res.Metrics
	t.Logf("total=%s sampling=%s steps=%d peakRSS=%dMB %s", m.Total, m.Sampling, m.Steps, m.PeakRSSBytes>>20, m.ModelVersion)
	if m.Total <= 0 || m.Sampling <= 0 || m.Steps != 4 || m.Backend != "cpu" {
		t.Fatalf("metrics are incomplete: %+v", m)
	}
	if len(m.Models) == 0 || m.Models[0].SHA256 == "" || m.Models[0].Quantization != "Q8_0" {
		t.Fatalf("model digests are incomplete: %+v", m.Models)
	}
	mu.Lock()
	defer mu.Unlock()
	if steps == 0 || last.Stage == "" {
		t.Fatalf("no progress was reported (steps=%d last=%+v)", steps, last)
	}
}

func TestSameSeedSamePicture(t *testing.T) {
	e := engine(t)
	req := diffusion.Request{Prompt: "a lighthouse at dusk", Width: 128, Height: 128, Seed: 7, Steps: 2, CFG: 5, Sampler: "euler"}
	a, err := e.Generate(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	b, err := e.Generate(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.PNG, b.PNG) {
		t.Fatal("the same seed and request gave two different pictures")
	}
	req.Seed = -1
	c, err := e.Generate(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if c.Seed < 0 {
		t.Fatalf("a random seed was reported as %d, which cannot be repeated", c.Seed)
	}
}

// TestInpaintKeepsTheMaskedPixels is the contract of inpainting: where the mask
// is black the picture is the original, pixel for pixel, and where it is white
// something was painted.
func TestInpaintKeepsTheMaskedPixels(t *testing.T) {
	e := engine(t)
	base := gradient(t, 256, 256)
	repaint := image.Rect(96, 96, 192, 176)
	mask, err := diffusion.EncodeMaskPNG(256, 256, repaint)
	if err != nil {
		t.Fatal(err)
	}
	res, err := e.Generate(context.Background(), diffusion.Request{
		Prompt: "a red barn", Init: base, Mask: mask, Strength: 1,
		Seed: 11, Steps: 4, CFG: 7, Sampler: "euler",
	})
	if err != nil {
		t.Fatalf("inpaint: %v", err)
	}
	if res.Mode != diffusion.ModeInpaint {
		t.Fatalf("mode = %s", res.Mode)
	}
	before, after := decode(t, base), decode(t, res.PNG)
	if after.Bounds() != before.Bounds() {
		t.Fatalf("size changed: %v -> %v", before.Bounds(), after.Bounds())
	}
	changedInside, changedOutside := 0, 0
	for y := 0; y < 256; y++ {
		for x := 0; x < 256; x++ {
			same := before.NRGBAAt(x, y) == after.NRGBAAt(x, y)
			if image.Pt(x, y).In(repaint) {
				if !same {
					changedInside++
				}
			} else if !same {
				changedOutside++
			}
		}
	}
	t.Logf("changed inside the mask: %d of %d; outside: %d", changedInside, repaint.Dx()*repaint.Dy(), changedOutside)
	if changedOutside != 0 {
		t.Fatalf("%d masked-out pixels changed; they must be identical", changedOutside)
	}
	if changedInside < repaint.Dx()*repaint.Dy()/4 {
		t.Fatalf("only %d pixels changed inside the repaint area: nothing was painted", changedInside)
	}
}

func TestCancelStopsTheRun(t *testing.T) {
	e := engine(t)
	ctx, cancel := context.WithCancel(context.Background())
	started := time.Now()
	_, err := e.Generate(ctx, diffusion.Request{
		Prompt: "a long run", Width: 128, Height: 128, Seed: 1, Steps: 40, Sampler: "euler",
		OnProgress: func(p diffusion.Progress) {
			if p.Stage == "sampling" && p.Completed >= 1 {
				cancel()
			}
		},
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	t.Logf("stopped after %s", time.Since(started))
	// The engine is usable afterwards: the cancellation did not leak into the
	// next run.
	if _, err := e.Generate(context.Background(), diffusion.Request{Prompt: "after", Width: 128, Height: 128, Seed: 1, Steps: 2, Sampler: "euler"}); err != nil {
		t.Fatalf("the run after a cancellation failed: %v", err)
	}
}

// TestReferenceEdit runs a FLUX.2 klein set when one is configured.
func TestReferenceEdit(t *testing.T) {
	set := diffusion.ModelSet{
		Diffusion: os.Getenv("DIFFUSION_FLUX2_DIFFUSION"),
		VAE:       os.Getenv("DIFFUSION_FLUX2_VAE"),
		LLM:       os.Getenv("DIFFUSION_FLUX2_LLM"),
	}
	if set.Diffusion == "" || set.VAE == "" || set.LLM == "" {
		t.Skip("DIFFUSION_FLUX2_DIFFUSION, _VAE and _LLM are not all set")
	}
	e, err := diffusion.Load(set, diffusion.WithThreads(threads()), diffusion.WithFlashAttention())
	if err != nil {
		t.Fatalf("loading the set: %v", err)
	}
	defer e.Close()
	src := gradient(t, 256, 256)
	res, err := e.Generate(context.Background(), diffusion.Request{
		Prompt: "make the picture warmer", References: [][]byte{src},
		Width: 256, Height: 256, Seed: 5, Steps: 4, CFG: 1, Sampler: "euler",
	})
	if err != nil {
		t.Fatalf("edit: %v", err)
	}
	if res.Mode != diffusion.ModeEdit {
		t.Fatalf("mode = %s", res.Mode)
	}
	decode(t, res.PNG)
	t.Logf("edit took %s, %s", res.Metrics.Total, res.Metrics.ModelVersion)
}
