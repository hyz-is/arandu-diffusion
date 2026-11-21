---
name: diffusion-package
description: Install, wire and use the Diffusion package (Go, Arandu) in an application -- image generation and editing with stable-diffusion.cpp in-process. Use when the request is to "install diffusion", "add image generation to the app", "go get github.com/hyz-is/arandu-diffusion", "wire it into bootstrap/app.go", "register the module", "generate an image", "inpaint", "edit with a reference image", "build with -tags sdcpp", "library stable-diffusion not found", "everything under /diffusion returns 403", "it redirects to /auth/login", "the table does not exist", or when a project's go.mod already requires github.com/hyz-is/arandu-diffusion. Covers the wiring, the build tag and archives, loading a model set, the request, the migration step, and why the policy refuses everything until the application opens it.
license: MIT
metadata:
  audience: app
---

# Using Diffusion

An Arandu package that is an image engine and a module: it links
stable-diffusion.cpp through cgo so the model runs in your process, and it keeps
what each run cost in its own table, behind its own routes and policy. It is
registered by hand -- there is no service provider, no container and no
discovery, so if a line is not written in the application, it does not happen.

## Install

```bash
go get github.com/hyz-is/arandu-diffusion@<latest tag>   # gh release list -R hyz-is/arandu-diffusion
```

Without a build tag the package compiles anywhere, and `Load` answers
`ErrBackendUnavailable`; `Fake` works. To run a model, build with `-tags sdcpp`:

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/hyz-is/arandu-diffusion
make -C arandu-diffusion archives                      # BACKEND=cpu|metal|cuda|vulkan
CGO_ENABLED=1 CGO_LDFLAGS="-L$PWD/arandu-diffusion/lib" go build -tags sdcpp ./...
```

A GPU backend adds a tag: `-tags "sdcpp sd_cuda"`. Do not link this and
arandu-llama into one binary (two copies of ggml): run diffusion in its own
worker.

## The wiring, and where each part goes

All of it is in `bootstrap/app.go`. The import, with the other module imports:

```go
import (
	diffusion "github.com/hyz-is/arandu-diffusion"
)
```

The engine is loaded once, where the application can fail at boot, and closed at
shutdown by whoever opened it. Paths come from configuration, never from a request:

```go
	engine, err := diffusion.Load(diffusion.ModelSet{
		Diffusion: cfg.Diffusion.Model, // FLUX.2 klein Q8_0 GGUF
		VAE:       cfg.Diffusion.VAE,
		LLM:       cfg.Diffusion.TextEncoder,
	}, diffusion.WithThreads(cfg.Diffusion.Threads))
```

The module, in `App`, once `NewAppServiceProvider` has returned. `New` refuses a
wiring that cannot work, so it is built where the error can leave:

```go
	diffusionModule, err := diffusion.New(diffusion.Config{
		Engine: engine,
		Policy: DiffusionRules{}, // your rules; see below
	}, drivers.DB, drivers.Sessions)
	if err != nil {
		_ = drivers.Close()
		return nil, err
	}
```

And the registration, in the `WithProviders(...)` chain:

```go
	return Configure(cfg, drivers, app).
		WithProviders(diffusionModule).
		WithVersion(version, commit).
		Create()
```

Then, once, before the application serves, `aru migrate` (the package owns the
`diffusions` table; `arandu.mod.toml` says `migrations = true`). Never at boot.

## Generate

```go
res, run, err := diffusionModule.Generate(ctx, subject, diffusion.Request{
	Prompt: "a lighthouse at dusk", Width: 512, Height: 512, Seed: -1, Steps: 20,
})
// res.PNG, res.Seed (the one to repeat it), res.Metrics; run is the kept row.
```

Inpaint: `Init` (PNG/JPEG, sides multiples of 8) plus `Mask` (white repaints,
black keeps; `diffusion.EncodeMaskPNG` builds a rectangle). **The masked pixels
come back identical**, and the guarantee is made in Go after the library. Edit
with references: `References: [][]byte{...}` on a model that supports it
(FLUX.2 klein, Kontext). The picture is returned as bytes: **store it in R2**, as
every file of an Arandu application. A long run belongs on your queue; call
`Generate` from the worker. `Record` keeps a run you made with the engine
yourself. In tests pass `diffusion.NewFake()`.

## The policy refuses everything until you open it

`DiffusionPolicy` has no branch that allows an action, so a correctly wired,
correctly migrated package answers 403 on day one. The application opens actions
with `Config.Policy` (it cannot edit the package): the rules, with the tenant
comparison kept and the actions filled by arandu-permission, are in
`diffusion-policy` and in the `Config.Policy` doc comment. Actions:
`diffusion.view`, `diffusion.list`, `diffusion.create` (the one that spends compute).

## Routes

| method | path | name |
| --- | --- | --- |
| `GET` | `/diffusion` | `diffusion.index` |
| `GET` | `/diffusion/{id}` | `diffusion.show` |
| `POST` | `/diffusion` | `diffusion.store` (only with `Config.Engine`; text-to-image, answers `image/png`) |

Every route sits behind `RequireAuth`: a request with no session is sent to
`/auth/login` (303), and returns to the page it asked for after signing in.
**Symptom:** every route redirects there -- nobody is signed in on that request,
or the session store passed to `New` is not the one the application signs in with.

| status | what happened |
| --- | --- |
| `303` to `/auth/login` | no session |
| `403` | the policy refused, or no rule allows the action yet |
| `404` | no run with that id in this tenant |
| `303` back / `422` | the input was rejected, messages keyed by field |

## What a run keeps, and what it does not

Who, mode, status, model name and digest, quantization, backend, threads, size,
steps, seed, sampler, cfg, durations and the process peak memory. Never the
picture, and **not the prompt unless `Config.KeepPrompts`** (its digest is always
kept). `TenantID` is in no response.

Issues and pull requests go to `github.com/hyz-is/arandu-diffusion`; a
vulnerability goes to the private advisory form in `SECURITY.md`. MIT licensed.
Copyright HYZIS.
