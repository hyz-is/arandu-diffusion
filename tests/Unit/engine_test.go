package unit_test

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	diffusion "github.com/hyz-is/arandu-diffusion"
)

// These tests hold the contract of Engine against the Fake, which is the same
// contract with no model. The real engine meets the same assertions in
// tests/Real, under the sdcpp tag, when a model is present.

func decodeNRGBA(t *testing.T, b []byte) *image.NRGBA {
	t.Helper()
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("not a png: %v", err)
	}
	out := image.NewNRGBA(img.Bounds())
	draw.Draw(out, out.Bounds(), img, img.Bounds().Min, draw.Src)
	return out
}

// photo is a picture every pixel of which differs from its neighbours, so a
// pixel that moved cannot hide.
func photo(t *testing.T, w, h int) []byte {
	t.Helper()
	src := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			src.SetNRGBA(x, y, color.NRGBA{R: uint8(x * 255 / w), G: uint8(y * 255 / h), B: uint8((x*7 + y*13) % 256), A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, src); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func generate(t *testing.T, e diffusion.Engine, req diffusion.Request) *diffusion.Result {
	t.Helper()
	res, err := e.Generate(context.Background(), req)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	return res
}

func TestTxt2ImgAnswersAPNGOfTheSizeAsked(t *testing.T) {
	t.Parallel()

	fake := diffusion.NewFake()
	var mu sync.Mutex
	var reports []diffusion.Progress
	res := generate(t, fake, diffusion.Request{
		Prompt: "a red apple", Width: 256, Height: 128, Seed: 42, Steps: 5,
		OnProgress: func(p diffusion.Progress) { mu.Lock(); reports = append(reports, p); mu.Unlock() },
	})

	img := decodeNRGBA(t, res.PNG)
	if img.Bounds().Dx() != 256 || img.Bounds().Dy() != 128 || res.Width != 256 || res.Height != 128 {
		t.Fatalf("size %v, result says %dx%d, want 256x128", img.Bounds(), res.Width, res.Height)
	}
	if res.Seed != 42 || res.Mode != diffusion.ModeTxt2Img {
		t.Fatalf("seed=%d mode=%s", res.Seed, res.Mode)
	}
	if res.Metrics.Steps != 5 || res.Metrics.Total <= 0 || res.Metrics.Backend != "fake" || len(res.Metrics.Models) != 1 {
		t.Fatalf("metrics are incomplete: %+v", res.Metrics)
	}
	if m := res.Metrics.Models[0]; m.SHA256 == "" || m.Quantization != "Q8_0" {
		t.Fatalf("the model is not identified: %+v", m)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(reports) != 5 || reports[4].Stage != "sampling" || reports[4].Completed != 5 || reports[4].Total != 5 {
		t.Fatalf("progress = %+v, want five sampling reports ending at 5/5", reports)
	}
	if got := fake.Requests(); len(got) != 1 || got[0].Prompt != "a red apple" {
		t.Fatalf("the fake recorded %+v", got)
	}
}

func TestTheDefaultsAreTheLibrarys(t *testing.T) {
	t.Parallel()

	res := generate(t, diffusion.NewFake(), diffusion.Request{Prompt: "x"})
	if res.Width != 512 || res.Height != 512 || res.Metrics.Steps != 20 {
		t.Fatalf("defaults are %dx%d at %d steps, want 512x512 at 20", res.Width, res.Height, res.Metrics.Steps)
	}
}

func TestTheSeedMakesARunRepeatable(t *testing.T) {
	t.Parallel()

	fake := diffusion.NewFake()
	req := diffusion.Request{Prompt: "a lighthouse", Width: 64, Height: 64, Seed: 7, Steps: 1}
	a, b := generate(t, fake, req), generate(t, fake, req)
	if !bytes.Equal(a.PNG, b.PNG) {
		t.Fatal("the same seed and prompt gave two pictures")
	}
	req.Seed = 8
	if c := generate(t, fake, req); bytes.Equal(a.PNG, c.PNG) {
		t.Fatal("another seed gave the same picture")
	}

	// A negative seed is random, is reported, and repeats the run when passed back.
	req.Seed = -1
	d := generate(t, fake, req)
	if d.Seed < 0 {
		t.Fatalf("a random seed was reported as %d", d.Seed)
	}
	req.Seed = d.Seed
	if e := generate(t, fake, req); !bytes.Equal(d.PNG, e.PNG) {
		t.Fatal("passing the reported seed back did not repeat the run")
	}
}

func TestARequestIsRefusedBeforeAnyComputeIsSpent(t *testing.T) {
	t.Parallel()

	good := photo(t, 64, 64)
	for name, req := range map[string]diffusion.Request{
		"no prompt":         {},
		"prompt too long":   {Prompt: strings.Repeat("a", diffusion.MaxPromptBytes+1)},
		"width not 8":       {Prompt: "x", Width: 100, Height: 64},
		"width too small":   {Prompt: "x", Width: 8, Height: 64},
		"height too big":    {Prompt: "x", Width: 64, Height: diffusion.MaxSide + 8},
		"negative width":    {Prompt: "x", Width: -64, Height: 64},
		"too many steps":    {Prompt: "x", Steps: diffusion.MaxSteps + 1},
		"negative steps":    {Prompt: "x", Steps: -1},
		"cfg out of range":  {Prompt: "x", CFG: 99},
		"strength above 1":  {Prompt: "x", Init: good, Strength: 1.5},
		"sampler not name":  {Prompt: "x", Sampler: "euler --evil"},
		"scheduler spaces":  {Prompt: "x", Scheduler: "a b"},
		"mask without init": {Prompt: "x", Mask: good},
		"too many refs":     {Prompt: "x", References: [][]byte{good, good, good, good, good}},
		"init not an image": {Prompt: "x", Init: []byte("not an image")},
		"mask not an image": {Prompt: "x", Init: good, Mask: []byte("nope")},
		"ref not an image":  {Prompt: "x", References: [][]byte{[]byte("nope")}},
		"mask other size":   {Prompt: "x", Init: good, Mask: photo(t, 72, 72)},
		"init not 8":        {Prompt: "x", Init: photo(t, 70, 70)},
		"size beside init":  {Prompt: "x", Init: good, Width: 128, Height: 128},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			fake := diffusion.NewFake(diffusion.Answer{})
			_, err := fake.Generate(context.Background(), req)
			if !errors.Is(err, diffusion.ErrInvalidRequest) {
				t.Fatalf("err = %v, want ErrInvalidRequest", err)
			}
		})
	}
}

func TestImg2ImgTakesItsSizeFromTheImage(t *testing.T) {
	t.Parallel()

	res := generate(t, diffusion.NewFake(), diffusion.Request{Prompt: "x", Init: photo(t, 96, 64), Steps: 1})
	if res.Mode != diffusion.ModeImg2Img || res.Width != 96 || res.Height != 64 {
		t.Fatalf("mode=%s size=%dx%d, want img2img at 96x64", res.Mode, res.Width, res.Height)
	}
}

func TestAJPEGIsAcceptedWhereAPNGIs(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, decodeNRGBA(t, photo(t, 64, 64)), nil); err != nil {
		t.Fatal(err)
	}
	res := generate(t, diffusion.NewFake(), diffusion.Request{Prompt: "x", Init: buf.Bytes(), Steps: 1})
	if res.Width != 64 {
		t.Fatalf("width = %d", res.Width)
	}
}

func TestEditTakesReferencesAndKeepsTheirOwnSizes(t *testing.T) {
	t.Parallel()

	res := generate(t, diffusion.NewFake(), diffusion.Request{
		Prompt: "make it warmer", References: [][]byte{photo(t, 70, 50), photo(t, 64, 64)},
		Width: 128, Height: 128, Steps: 1,
	})
	if res.Mode != diffusion.ModeEdit || res.Width != 128 {
		t.Fatalf("mode=%s width=%d", res.Mode, res.Width)
	}
}

// TestInpaintKeepsTheMaskedPixelsIdentical is the contract of inpainting, and
// the one this package makes itself rather than takes from the library.
func TestInpaintKeepsTheMaskedPixelsIdentical(t *testing.T) {
	t.Parallel()

	const size = 128
	original := photo(t, size, size)
	repaint := image.Rect(32, 40, 96, 100)
	mask, err := diffusion.EncodeMaskPNG(size, size, repaint)
	if err != nil {
		t.Fatal(err)
	}
	res := generate(t, diffusion.NewFake(), diffusion.Request{Prompt: "a red barn", Init: original, Mask: mask, Seed: 1, Steps: 2})
	if res.Mode != diffusion.ModeInpaint {
		t.Fatalf("mode = %s", res.Mode)
	}

	before, after := decodeNRGBA(t, original), decodeNRGBA(t, res.PNG)
	insideChanged, outsideChanged := 0, 0
	for y := range size {
		for x := range size {
			same := before.NRGBAAt(x, y) == after.NRGBAAt(x, y)
			switch {
			case image.Pt(x, y).In(repaint) && !same:
				insideChanged++
			case !image.Pt(x, y).In(repaint) && !same:
				outsideChanged++
			}
		}
	}
	if outsideChanged != 0 {
		t.Fatalf("%d pixels outside the mask changed", outsideChanged)
	}
	if insideChanged < repaint.Dx()*repaint.Dy()*9/10 {
		t.Fatalf("only %d of %d pixels inside the mask changed", insideChanged, repaint.Dx()*repaint.Dy())
	}
}

func TestAGreyMaskBlendsAndAnAlphaMaskKeepsWhatItDoesNotCover(t *testing.T) {
	t.Parallel()

	const size = 64
	original := photo(t, size, size)

	// Half-grey on the left, transparent white on the right: grey blends,
	// transparent is read as black and keeps.
	m := image.NewNRGBA(image.Rect(0, 0, size, size))
	for y := range size {
		for x := range size {
			if x < size/2 {
				m.SetNRGBA(x, y, color.NRGBA{R: 128, G: 128, B: 128, A: 255})
			} else {
				m.SetNRGBA(x, y, color.NRGBA{R: 255, G: 255, B: 255, A: 0})
			}
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, m); err != nil {
		t.Fatal(err)
	}
	res := generate(t, diffusion.NewFake(), diffusion.Request{Prompt: "x", Init: original, Mask: buf.Bytes(), Seed: 3, Steps: 1})
	before, after := decodeNRGBA(t, original), decodeNRGBA(t, res.PNG)
	for y := range size {
		for x := size / 2; x < size; x++ {
			if before.NRGBAAt(x, y) != after.NRGBAAt(x, y) {
				t.Fatalf("the transparent part of the mask changed pixel %d,%d", x, y)
			}
		}
	}
	blended := 0
	for y := range size {
		for x := range size / 2 {
			if before.NRGBAAt(x, y) != after.NRGBAAt(x, y) {
				blended++
			}
		}
	}
	if blended == 0 {
		t.Fatal("a mid-grey mask changed nothing")
	}
}

func TestACancelledContextStopsTheRun(t *testing.T) {
	t.Parallel()

	fake := diffusion.NewFake()
	fake.Delay = 20 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	started := time.Now()
	var once sync.Once
	_, err := fake.Generate(ctx, diffusion.Request{
		Prompt: "x", Width: 64, Height: 64, Steps: 100,
		OnProgress: func(diffusion.Progress) { once.Do(cancel) },
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if time.Since(started) > time.Second {
		t.Fatalf("the run took %s to stop", time.Since(started))
	}

	done, stop := context.WithCancel(context.Background())
	stop()
	if _, err := fake.Generate(done, diffusion.Request{Prompt: "x"}); !errors.Is(err, context.Canceled) {
		t.Fatalf("an already-cancelled context was not refused: %v", err)
	}
}

func TestAScriptedAnswerIsSpentThenTheFakeGenerates(t *testing.T) {
	t.Parallel()

	boom := errors.New("out of memory")
	fake := diffusion.NewFake(diffusion.Answer{Err: boom})
	if _, err := fake.Generate(context.Background(), diffusion.Request{Prompt: "x"}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the scripted error", err)
	}
	generate(t, fake, diffusion.Request{Prompt: "x", Width: 64, Height: 64, Steps: 1})
	fake.Push(diffusion.Answer{Err: boom})
	if _, err := fake.Generate(context.Background(), diffusion.Request{Prompt: "x"}); !errors.Is(err, boom) {
		t.Fatalf("a pushed answer was not given: %v", err)
	}
	if got := len(fake.Requests()); got != 3 {
		t.Fatalf("the fake recorded %d requests, want 3", got)
	}
}

func TestAClosedEngineSaysSo(t *testing.T) {
	t.Parallel()

	fake := diffusion.NewFake()
	if err := fake.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := fake.Generate(context.Background(), diffusion.Request{Prompt: "x"}); !errors.Is(err, diffusion.ErrClosed) {
		t.Fatalf("err = %v, want ErrClosed", err)
	}
}

func TestAModelSetNamesOneModelAndFilesThatExist(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	file := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	for name, set := range map[string]diffusion.ModelSet{
		"empty":                {},
		"both models":          {Checkpoint: file, Diffusion: file},
		"missing file":         {Checkpoint: filepath.Join(dir, "absent.gguf")},
		"a directory":          {Checkpoint: dir},
		"missing vae":          {Diffusion: file, VAE: filepath.Join(dir, "absent.safetensors")},
		"lora without path":    {Checkpoint: file, LoRAs: []diffusion.LoRA{{}}},
		"lora scale too large": {Checkpoint: file, LoRAs: []diffusion.LoRA{{Path: file, Scale: 50}}},
	} {
		if err := set.Validate(); !errors.Is(err, diffusion.ErrInvalidModelSet) {
			t.Errorf("%s: err = %v, want ErrInvalidModelSet", name, err)
		}
	}
	good := diffusion.ModelSet{Diffusion: file, VAE: file, LLM: file, LoRAs: []diffusion.LoRA{{Path: file, Scale: 0.5}}}
	if err := good.Validate(); err != nil {
		t.Fatalf("a complete set was refused: %v", err)
	}
}

func TestLoadWithoutTheTagSaysHowToGetTheEngine(t *testing.T) {
	t.Parallel()

	if diffusion.Linked {
		t.Skip("this build links stable-diffusion.cpp")
	}
	dir := t.TempDir()
	file := filepath.Join(dir, "model.gguf")
	if err := os.WriteFile(file, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := diffusion.Load(diffusion.ModelSet{Checkpoint: file}); !errors.Is(err, diffusion.ErrBackendUnavailable) {
		t.Fatalf("err = %v, want ErrBackendUnavailable", err)
	}
	// A set that is wrong is reported as wrong first: the tag is not the
	// caller's problem when the paths are.
	if _, err := diffusion.Load(diffusion.ModelSet{}); !errors.Is(err, diffusion.ErrInvalidModelSet) {
		t.Fatalf("err = %v, want ErrInvalidModelSet", err)
	}
	if _, err := diffusion.Load(diffusion.ModelSet{Checkpoint: file}, diffusion.WithThreads(-1)); !errors.Is(err, diffusion.ErrInvalidModelSet) {
		t.Fatalf("a negative thread count was accepted: %v", err)
	}
}

func TestOptionsReachLoad(t *testing.T) {
	t.Parallel()

	// The options are plain functions; this holds that they are accepted in any
	// number and order, and that none of them is nil-unsafe.
	opts := []diffusion.Option{
		diffusion.WithThreads(4), diffusion.WithBackend("cpu"), diffusion.WithFlashAttention(),
		diffusion.WithoutDigests(), diffusion.WithLogger(func(string, string) {}),
	}
	if _, err := diffusion.Load(diffusion.ModelSet{}, opts...); !errors.Is(err, diffusion.ErrInvalidModelSet) {
		t.Fatalf("err = %v", err)
	}
}
