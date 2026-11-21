---
name: diffusion-module
description: Add or change a route, a handler, a config field, a response field, a column or a migration of this Arandu package. Use when the request is to "add an endpoint", "add a field to the run", "add a column", "change the default prefix", "expose X in the response", "write a migration", "add a Config option", "regenerate DiffusionQuery", or when a change touches module.go, config.go, model.go or DiffusionQuery.go.
license: MIT
---

# Changing the module half

`foundation.Module` is `Name()` and `Routes()`, and nothing else. This package
adds `Migrations()` (`foundation.Migratable`) because it owns a table. It
implements no other optional interface: the engine it uses belongs to the
application, which closes it.

## Routes

```go
r = r.Group("", middleware.RequireAuth(m.sessions))
r.Action(stdhttp.MethodGet, m.cfg.Prefix, m.index).Name("diffusion.index")
```

- The guard is mounted once, on the router, before the first route. A route
  registered outside `RequireAuth` is a security failure and a failing test.
- The generating route (`POST`) is registered **only when `Config.Engine` is
  set**. A route that could only answer "not configured" exists to be probed.
- Names are the public contract: URLs are built from them, so a rename is a 404
  in somebody else's template. Say so in the changelog.
- A prefix arrives through configuration, so `/_arandu/` is checked by reading
  the route table back (`TestNoRouteLandsInTheFrameworkNamespace`).

## Handlers

Thin: read the input, ask the service, answer. A handler reads who is asking with
`ctx.User()` (the subject `RequireAuth` put on the request), passes it through
`m.acting` (which honours `Config.Subject`), calls one service method and **`return err`**
-- the router answers it: 403 for a refusal, 404 for `ErrNotFound`, a flash or a
422 for `validation.Errors`. A handler never loads the session, never writes a
status for an error, never inspects one with `errors.Is`. A request the engine
would refuse (`ErrInvalidRequest`) reaches the router as `validation.Errors`
through `asValidation`, in the service.

Input is parsed by field and a bad number is a rejected input naming the field,
never a default: a width the client wrote and did not get is worse than one it
was told about.

## The entity and its table

`Diffusion` embeds `model.Model`; `diffusionTable`, beside it, declares the table
once (`ManualKey: true`, because the text primary key is written by the
application). `aru model:build` generates `DiffusionQuery.go` from it: run it
after any change to the struct, and commit the file. `aru model:build --check`
exits 1 when it is stale.

Rules of the row itself -- an invariant, a derived value -- go in the **custom
block** of `model.go`, as pure methods. Orchestration stays in the service.

A new column is four edits in one commit: the field in `model.go`, the column in
`Migrations()` (a **new migration** if the table has shipped -- the name of an
applied migration is fixed), the field in `Resource.ToArray` if it may leave, and
`fillRun` if the run supplies it. `TenantID` is in no response.

## Config

Typed struct, validated by `New`, defaults applied after validation, never
before. A field that cannot work is an error from `New`, not a panic and not a
clamp. `Config.Engine`, `Config.Policy` and `Config.Subject` are the three the
application supplies; **the engine is the application's, and it closes it**.
There is no tenant in `Config`, because every tenant comes from the Grant.

## Migrations

`createDiffusions` is reversible and dated in its name; the Blueprint spells each
column for the engine it runs on, so the same migration serves SQLite and
Postgres. The suite applies the module's own declaration to SQLite in memory
(`service_internal_test.go`), so the schema the tests exercise is the schema an
installer gets.
