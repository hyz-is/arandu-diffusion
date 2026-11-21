---
name: diffusion-engine
description: Work on the cgo engine of this package -- the stable-diffusion.cpp submodule, the archives, the mirrored header, the build tags, a link failure, the progress and cancel plumbing, the inpainting mask guarantee, the Fake, the real-model test. Use when the request is to "bump stable-diffusion.cpp", "build the archives", "ld: library not found", "undefined symbol sd_", "add a backend", "the model does not load", "the mask changed pixels", "add a request field", "cancel does not stop", or when a change touches engine.go, engine_sdcpp.go, imaging.go, modelfile.go, fake.go, link_*.go, the Makefile or cgo_headers/.
license: MIT
---

# The engine, and where it can hurt

## Two builds, two failures

The **Go half** compiles anywhere with no C toolchain: without the `sdcpp` tag,
`Load` answers `ErrBackendUnavailable` and everything else -- the request, the
imaging, the digests, the `Fake`, the module -- is plain Go. `go vet ./...` proves
it, and it is what the four gates run.

The **linked half** is `-tags sdcpp`. It needs the header (mirrored in
`cgo_headers/`, so `go vet -tags sdcpp ./...` works with no archives) and, to
link, the archives in `lib/`:

```sh
git submodule update --init stable-diffusion.cpp
git -C stable-diffusion.cpp submodule update --init ggml
make archives                 # BACKEND=cpu|metal|cuda|vulkan
CGO_ENABLED=1 go test -tags sdcpp -count=1 -v ./tests/Real/
```

`ld: library 'stable-diffusion' not found` is not a compile error: the cgo
already compiled. The archives are missing, or `-L` does not name them.

On this Mac the Command Line Tools SDK has a broken `libSystem.tbd` for the
linker; build and test with
`SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.

## The pin, and the mirrored header

`stable-diffusion.cpp/` is a submodule at one commit, and its nested `ggml` is
the only nested submodule the build needs (WebP and WebM are off). The Go module
proxy strips submodules from the module zip, so **the one public header is
mirrored byte for byte in `cgo_headers/stable-diffusion.h`**; CI runs
`make check-headers`. A bump is: move the submodule, `make headers`, rebuild the
archives, run the real test, record the commit in the changelog. The C API is
string-keyed (`sd_opt_set_string(opt, "sample.steps", ...)`), documented in
`stable-diffusion.cpp/docs/guides/c_api.md`; an unknown key is
`SD_STATUS_UNKNOWN_KEY` at the setter, which is why a new request field is one
line in `configure` and one test.

Backends add a tag on top of `sdcpp`: `sd_metal`, `sd_cuda`, `sd_vulkan`. They do
not reuse `metal` or `cublas`, which arandu-llama owns. **Do not link this engine
and arandu-llama into one binary**: both carry their own ggml, the symbols
collide. Run diffusion in its own process or worker.

## Rules that keep it honest

- **One OS thread per call.** `sd_get_last_error` is thread-local, and a goroutine
  moves between cgo calls. `Load` and `Generate` `runtime.LockOSThread()` for the
  length of their calls and read the diagnostic on that thread.
- **One generation at a time.** The semaphore is a channel so a waiter gives up
  with its context. `Close` cancels, then takes the semaphore, then frees.
- **Cancellation is joined.** The watcher goroutine is waited for before
  `SD_CANCEL_RESET`, or a late cancel lands on the next run.
- **A callback must not panic through C.** `goDiffusionProgress` recovers.
- **Progress comes from the context-local callback.** The legacy global callback
  is registered only as a no-op, because without one the library draws a text bar
  on stderr.
- **The mask guarantee is made in Go.** The library keeps the masked area close to
  the original, not equal, because the whole image goes through the lossy VAE.
  `composite` copies original bytes where the mask is 0. Never "fix" a mask test
  by loosening it.
- **The reported seed is the one used.** A negative seed is drawn in Go
  (`randomSeed`) and passed explicitly, so the run can be repeated.
- **No prompt in an error or a log line kept.** Library diagnostics are struck of
  the prompt before they enter an error (`recentLogs`, `shorten`).
- **Paths are configuration, never request input.** A `Request` names no file.
- **Model files are digested at `Load`**, concurrently, and the quantization is the
  type holding the most parameters in the GGUF header (`modelfile.go`), not the
  most tensors.

## Tests

`tests/Unit/engine_test.go` holds the contract against the `Fake`; the root's
`*_internal_test.go` hold the GGUF reader, the digests and the service against
SQLite in memory. `tests/Real` runs the library and skips itself, saying why,
unless `DIFFUSION_MODEL` names a file:

```sh
DIFFUSION_MODEL=~/.cache/hyzis-dev/ferflow/models/sd15-inpaint-Q8_0.gguf \
  CGO_ENABLED=1 go test -tags sdcpp -count=1 -v ./tests/Real/
```

A FLUX.2 klein set (`DIFFUSION_FLUX2_DIFFUSION`, `_VAE`, `_LLM`) enables the
reference-image edit. Timings on a loaded machine mean nothing; the assertions
are on contract, not on speed.
