# Changelog

Everything worth knowing about a release of Diffusion is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

## [0.1.0] - 2026-11-20

First release.

### Added

- The engine: `Load(ModelSet, ...Option)` returns an `Engine` over
  stable-diffusion.cpp (pinned at `532d2c1`) linked through cgo, behind the
  `sdcpp` build tag; without the tag `Load` answers `ErrBackendUnavailable` and the
  rest of the package compiles anywhere. A GPU backend adds `sd_metal`, `sd_cuda`
  or `sd_vulkan`; `make archives BACKEND=...` builds the static archives from the
  submodule, and `cgo_headers/` mirrors its one public header.
- `Engine.Generate`: txt2img, img2img, inpainting under a mask and
  reference-image editing, with seed, steps, CFG, distilled guidance, sampler,
  scheduler, clip skip, width and height; a progress callback; cancellation
  through the context; PNG out.
- Masked pixels come back identical: the original is composited back where the
  mask is black, since the library alone keeps them close but not equal.
- `Metrics` on every `Result`: wall time, sampling time, steps, the process peak
  memory, backend, threads, model family, and for each model file its SHA-256 and
  the quantization read from the GGUF header.
- `ModelSet` (checkpoint or diffusion model, VAE, CLIP-L, CLIP-G, T5-XXL, LLM,
  LoRAs), `Option`s `WithThreads`, `WithBackend`, `WithFlashAttention`,
  `WithoutDigests`, `WithLogger`, and `Fake`, the `Engine` for tests.
- The module: `New`, `Config` (`Prefix`, `PageSize`, `Engine`, `KeepPrompts`,
  `Policy`, `Subject`), `Module.Generate` and `Module.Record`.
- The `Diffusion` entity, one run kept, in the `diffusions` table
  (migration `20261120_0001_create_diffusions`), and the actions `DiffusionView`,
  `DiffusionList` and `DiffusionCreate`. The policy denies everything until the
  application passes its own in `Config.Policy`.
- Routes `diffusion.index`, `diffusion.show` and, only with an engine,
  `diffusion.store` (text-to-image, answers `image/png`), behind `RequireAuth`.
- `tests/Real`, behind the `sdcpp` tag and `DIFFUSION_MODEL`, which skips itself
  when the model is absent.

