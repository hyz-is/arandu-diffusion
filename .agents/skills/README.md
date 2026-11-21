# Skills

Procedures an assistant follows when working on this package.

They live in `.agents/skills/<name>/SKILL.md`, which is the path the coding
assistants read from. Each file opens with frontmatter carrying a `name` and a
`description`; the `name` has to equal the directory name exactly, or the skill
is not loaded. The description names the situation you are in rather than the
topic it covers.

| skill | when it fires |
| --- | --- |
| `diffusion-engine` | cgo, the submodule, the archives, the mirrored header, a link failure, the mask guarantee, the Fake |
| `diffusion-policy` | opening an action, an authorized use case, anything answering 403 |
| `diffusion-module` | adding a route, a handler, a config field, a response field or a migration |
| `diffusion-release` | the gates, the manifest, a dependency, a version, a tag |
| `diffusion-vault-notes` | writing the note, the gap or the journal entry, when this checkout sits inside the Arandu Obsidian vault |
| `diffusion-package` | installing and wiring this package **into an application** |

`diffusion-package` has a different audience from the others, on purpose: it
travels with the package so that an assistant working in somebody else's
project -- the one running `go get` -- has the wiring, the migration step, the
build tag and the closed policy in front of it instead of guessing. Its
frontmatter says so under `metadata`, with `audience: app`, which is what
`aru skills:sync` reads; no other skill here carries it.

`diffusion-vault-notes` fires on a condition rather than on a task: it applies
only when `MOC-arandu.md`, `plans/cmd/audit-vault/` and `45-modules/` are
actually beside this checkout. Outside the vault it is inert.

## Why these exist

The package has two halves that fail in different ways. The engine half is where
a mistake **crashes or is believed**: a header that drifted from the archives, a
progress callback that panics through C frames, a mask that is "close enough" and
is not pixel-identical, a seed that is not the one reported. The module half is
where a mistake **leaks**: a tenant read from the URL, a policy branch that
returns nil "for now", a prompt stored by default, a handler that picks a status
itself, a rule of the entity written into the service instead of the custom block
of `model.go`, a hand edit to the generated `DiffusionQuery.go`.

The package is built to be checked rather than trusted. Its denial tests use a
nil database, so reaching even the generated query constructor before
authorization panics. The structural twin reads every exported service method
and checks the same order on the allowed path, and reads every handler for the
subject and the status it must leave to the framework. The engine's contract is
asserted against the Fake in every run and against the real library whenever a
model is present. `go test -race ./...` exercises all of it.
