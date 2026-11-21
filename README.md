<p align="center">
  <img src=".github/logo.svg" alt="Arandu" width="180">
</p>

# Diffusion

Image generation and editing inside your own process. It links
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) through
cgo, so the model runs in the application rather than behind an HTTP call, on the
CPU with GGUF-quantized weights today and on a GPU in a cluster later. It is an
Arandu package, so it also owns a table where each run is kept, registers its own
routes and decides for itself who may reach either.

What it does:

- **text to image**, **image to image**, **inpainting** under a mask, and
  **reference-image editing** on the models that have it (FLUX.2 klein, Kontext);
- **masked pixels come back identical**: where the mask is black the result is the
  original, byte for byte;
- seed, steps, CFG, sampler, scheduler, width and height; a progress callback and
  cancellation through the context;
- PNG out, and the **metrics of every run**: time, the process's peak memory, the
  SHA-256 of each model file and the quantization read from the GGUF;
- a `Fake` with the same contract, so the code that generates is tested without a
  model or a C toolchain.

It needs `CGO_ENABLED=1` and the archives below only for the real engine. Without
the `sdcpp` build tag the package compiles anywhere, `Load` answers
`ErrBackendUnavailable`, and everything else -- the `Fake`, the module, the
recorded runs -- works.

## Install

```bash
go get github.com/hyz-is/arandu-diffusion@<latest tag>   # gh release list -R hyz-is/arandu-diffusion
```

### Build the archives

The module ships no compiled code. The archives are produced from the pinned
`stable-diffusion.cpp` submodule (`532d2c1`), once:

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/hyz-is/arandu-diffusion
cd arandu-diffusion
make archives                          # CPU, the default
make archives BACKEND=metal            # Apple silicon
make archives BACKEND=cuda CUDA_ARCHITECTURES=75
make archives BACKEND=vulkan
```

`lib/` then holds `libstable-diffusion.a` and the ggml archives of the backend.
Point the linker at them from the application, and build with the tag:

```bash
CGO_ENABLED=1 CGO_LDFLAGS="-L/path/to/arandu-diffusion/lib" go build -tags sdcpp ./...
```

The second `-L` is the whole trick: the package declares `-L${SRCDIR}/lib`, which
for a dependency is the read-only module cache, and the linker searches every
directory it is given. A GPU backend adds a tag on top: `-tags "sdcpp sd_metal"`,
`sd_cuda` or `sd_vulkan`. Pick the CUDA architecture of the card -- a build for a
higher one loads and dies at the first kernel launch.

**Do not link this engine and arandu-llama into one binary.** Each carries its own
copy of ggml and the symbols collide. Run diffusion in its own process, which is
also how a GPU worker in a cluster is shaped.

The Go module proxy strips git submodules, so the one public header the cgo needs
is mirrored in `cgo_headers/` and CI checks it against the submodule's.

## Wire it

An Arandu application registers a module explicitly. There is no service
provider, no container and no discovery, so these are the lines to paste into
`bootstrap/app.go`.

The import, with the other module imports:

```go
import (
	diffusion "github.com/hyz-is/arandu-diffusion"
)
```

Load the engine once, where the application can fail at boot. The paths come from
configuration, never from a request, and whoever opens the engine closes it:

```go
	engine, err := diffusion.Load(diffusion.ModelSet{
		Diffusion: cfg.Diffusion.Model,       // flux-2-klein-4b-Q8_0.gguf
		VAE:       cfg.Diffusion.VAE,         // flux2-vae.safetensors
		LLM:       cfg.Diffusion.TextEncoder, // qwen3-4b-Q8_0.gguf
	}, diffusion.WithThreads(8))
	if err != nil {
		return nil, err
	}
```

The module, in `App`, once `NewAppServiceProvider` has returned. `New` refuses a
wiring that cannot work and `Configure` has no way to return that error, so the
module is built where the error can leave; the database and the session store are
the ones the framework built from the configuration:

```go
	diffusionModule, err := diffusion.New(diffusion.Config{
		Engine: engine,
		Policy: DiffusionRules{}, // see "Open the policy"
	}, drivers.DB, drivers.Sessions)
	if err != nil {
		_ = drivers.Close()
		return nil, err
	}
```

And the registration, in the `WithProviders(...)` chain the builder is made of:

```go
	return Configure(cfg, drivers, app).
		WithProviders(diffusionModule).
		WithVersion(version, commit).
		Create()
```

Then, once, before the application serves:

```bash
aru migrate
```

This package owns the `diffusions` table, which is why the migration step is not
optional and why `arandu.mod.toml` says `migrations = true`.

## Generate

```go
res, run, err := diffusionModule.Generate(ctx, subject, diffusion.Request{
	Prompt: "a lighthouse at dusk", Width: 512, Height: 512,
	Seed: -1, Steps: 20, CFG: 7, Sampler: "euler",
	OnProgress: func(p diffusion.Progress) { /* p.Stage, p.Completed, p.Total */ },
})
// res.PNG      the picture
// res.Seed     the seed that was used: pass it back to repeat the run
// res.Metrics  time, peak memory, model digests and quantization
// run          the kept row of the diffusions table
```

`Generate` asks the policy for `diffusion.create` **before** any compute is spent,
runs the engine, and keeps the run -- failed and cancelled ones too, since they
cost the same. Without the module, use the engine directly:
`engine.Generate(ctx, req)`.

Inpainting keeps what the mask keeps. White is repainted, black is kept, grey
blends, and a transparent pixel is read as black:

```go
mask, _ := diffusion.EncodeMaskPNG(512, 512, image.Rect(128, 128, 384, 320))
res, _, err := diffusionModule.Generate(ctx, subject, diffusion.Request{
	Prompt: "a red barn", Init: original, Mask: mask, Strength: 1, Seed: 11, Steps: 20,
})
```

Editing with reference images, on a model that supports it:

```go
res, _, err := diffusionModule.Generate(ctx, subject, diffusion.Request{
	Prompt: "put the shoe on a wooden table", References: [][]byte{shoePNG},
	Width: 1024, Height: 1024, Steps: 4, CFG: 1,
})
```

Images are PNG or JPEG bytes the application already holds; the **picture is
returned as bytes and the application stores it in R2**, like every file. A run
takes minutes on a CPU: put it on the application's queue and call `Generate`
from the worker. `Module.Record` keeps a run made with the engine directly.

In tests, `diffusion.NewFake()` keeps the whole contract -- validation, progress,
cancellation, the mask guarantee, seeds -- and paints a pattern derived from the
seed and the prompt.

### Model sets

| family | `ModelSet` |
| --- | --- |
| SD 1.x, 2.x, SDXL | `Checkpoint` (carries the encoders and the VAE) |
| FLUX.1 | `Diffusion`, `VAE`, `CLIPL`, `T5XXL` |
| FLUX.2 klein | `Diffusion`, `VAE`, `LLM` (Qwen3) |
| Z-Image, Qwen-Image | `Diffusion`, `VAE`, `LLM` |

`LoRAs` are applied to every run. `Load` digests every file (SHA-256, concurrently)
and reads the quantization from the GGUF header, so each run says which exact
weights made it; `WithoutDigests` skips the hashing in tests.

`Metrics.PeakRSSBytes` is the process's high-water mark when the run finished: it
covers the whole process since it started, the model load included, so it bounds
the run from above and does not isolate it.

## Configuration

| field | required | meaning |
| --- | --- | --- |
| `Prefix` | no | where the routes are mounted. Defaults to `/diffusion`. |
| `PageSize` | no | how many runs one page answers with. Defaults to 25, refused above 200. |
| `Engine` | no | what generates. Without it the module keeps and lists runs, and the generating route is not registered. |
| `Policy` | no | who may do what. Nil is the shipped policy, which denies everything. |
| `Subject` | no | who is acting, for an application whose subject is more than the session (a member inside the organization they selected). |
| `KeepPrompts` | no | store the prompt text with its run. Off by default: the digest of the prompt is always kept. |

The zero `Config{}` is a valid configuration. There is no tenant to set: every
route needs a session, and the tenant of every statement is the one on the Grant
of whoever is signed in.

## Routes

| method | path | name |
| --- | --- | --- |
| `GET` | `/diffusion` | `diffusion.index` |
| `GET` | `/diffusion/{id}` | `diffusion.show` |
| `POST` | `/diffusion` | `diffusion.store` |

`POST` is registered only when `Config.Engine` is set, takes the text-to-image
fields (`prompt`, `negative_prompt`, `width`, `height`, `seed`, `steps`, `cfg`,
`sampler`, `scheduler`) and answers `image/png` with the run's id and the seed in
`X-Diffusion-Run` and `X-Diffusion-Seed`. An image arriving over HTTP would be a
second place a file enters the application, so the other modes are
`Module.Generate`, called from the handler that read the image from storage.

Every one of them sits behind `RequireAuth`: a visitor with no session is sent
to `/auth/login`, and the handler of a request that has one reads who is asking
with `ctx.User()`. Behind the guard, every route is refused until the policy is
opened. A handler returns the error the service returned, and the router answers
it: a refusal is 403, a run that is not there is 404, a rejected input goes back
with the messages in the flash, or is a 422 keyed by field for a client that asks
for JSON.

## Open the policy

The shipped `DiffusionPolicy` denies every action and has no branch that allows
one. An application cannot edit it after `go get`, so it passes its own rules in
`Config.Policy`, beside the rest of its authorization:

```go
type DiffusionRules struct{}

func (DiffusionRules) Can(_ context.Context, s security.Subject, a security.Action, record diffusion.Diffusion) error {
	if record.ID != "" && record.TenantID != s.Tenant {
		return fmt.Errorf("the run belongs to another tenant")
	}
	if !s.Can(a) { // the actions arandu-permission put on the subject
		return fmt.Errorf("the subject does not carry %s", a)
	}
	return nil
}

func (DiffusionRules) Writes(a security.Action) bool { return diffusion.DiffusionPolicy{}.Writes(a) }
```

| action | what it is |
| --- | --- |
| `diffusion.view` | reading one run |
| `diffusion.list` | paging through the runs |
| `diffusion.create` | running the engine and keeping the run; the one that spends compute |

Keep the tenant comparison whatever else is written: every rule below it holds
across customers only with it. Roles, groups and operator access come from
`hyz-is/arandu-permission`; this package never grows a role table.

## The data path

`Diffusion` embeds `model.Model`, and `diffusionTable`, beside it in `model.go`,
declares its table once. `aru model:build` generates `DiffusionQuery.go` from it:
`Diffusions(db)`, which starts a `*DiffusionQuery`, and `DiffusionCollection`. The
file is committed and never edited by hand; regenerate it after changing the
entity, and `aru model:build --check` fails when it is out of date.

`DiffusionService` owns `*data.DB` and follows
`validate -> security.Authorize -> Grant -> engine -> Model terminal`; handlers
never hold the database or start a query. The row is built with
`Diffusions(s.db).New()` and its tenant is `data.Tenant(g)`. A run keeps who asked,
the mode, how it ended, the model name, digest and quantization, the backend, the
knobs (the seed being the one used) and what it cost. Never the picture. There is
no CRUD Repository.

## Layout

```
engine.go          Engine, ModelSet, Request, Result, Metrics, the errors and the limits
engine_sdcpp.go    the cgo engine (tag sdcpp); engine_stub.go without it
link_*.go          linker flags per backend: cpu, sd_metal, sd_cuda, sd_vulkan
fake.go            Fake, the Engine for tests
imaging.go         decoding, masks, the composite that keeps masked pixels
modelfile.go       digests and the quantization read from a GGUF header
module.go          registration, routes, handlers and migrations
config.go          what the application passes in
model.go           the entity, its table, and what it may answer with
DiffusionQuery.go  the query aru model:build generates from model.go
policy.go          who may do what
service.go         the use cases, and Model access after authorization
stable-diffusion.cpp/  the pinned submodule
cgo_headers/       the one public header, mirrored byte for byte
```

## Tests

```bash
go test -race ./...                    # the gates; no model, no C toolchain
```

The denial suite constructs the service with a nil database, so even building
`Diffusions(nil)` would panic; the structural twin reads every service method for
`Authorize` before the model; the engine's contract is asserted against the Fake;
the service runs against SQLite in memory.

The real library is exercised by one test behind a build tag and an environment
variable. It skips itself, saying why, when the model is absent:

```bash
DIFFUSION_MODEL=~/.cache/hyzis-dev/ferflow/models/sd15-inpaint-Q8_0.gguf \
  CGO_ENABLED=1 go test -tags sdcpp -count=1 -v ./tests/Real/
```

It generates a 256x256 picture on the CPU from an SD 1.5 Q8_0 checkpoint, checks
the metrics and digests, repeats a seed, cancels a run, and proves that inpainting
leaves the masked pixels unchanged. With `DIFFUSION_FLUX2_DIFFUSION`,
`DIFFUSION_FLUX2_VAE` and `DIFFUSION_FLUX2_LLM` set it also runs a FLUX.2 klein
reference edit.

## Licence

MIT. See [LICENSE.md](LICENSE.md). Copyright HYZIS.

`stable-diffusion.cpp/` is a submodule of
[leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp),
MIT, with its own notice, and so is its ggml. `cgo_headers/stable-diffusion.h` is
that project's header, mirrored unchanged. Nothing here says anything about the
licence on model weights, which is per model and the application's to check.
