# Security policy

## Supported versions

The latest minor release of Diffusion receives security fixes. Older minor
releases do not: a published Go module version is immutable, so a fix is a new
release rather than a moved tag.

## Reporting a vulnerability

Report it privately, through GitHub's advisory form:

<https://github.com/hyz-is/arandu-diffusion/security/advisories/new>

Do not open a public issue and do not describe the problem in a pull request.
A report that arrives in public is a report every reader of this repository can
act on before there is a release to upgrade to.

Include what you need to reproduce it: the version, the wiring, and the request
or call that triggers it.

Expect an acknowledgement within a few days. If the report is confirmed, the
fix, the release and the advisory are published together, and you are credited
unless you ask not to be.

## What is in scope

Anything that lets a caller reach data a policy did not authorize. In
particular:

- a path from a handler to a Model or database handle that does not pass through
  the Service and its Policy;
- a route reachable without `RequireAuth`, or a handler that takes who is
  asking from anywhere but the subject the guard put on the request;
- a Model terminal reached before `security.Authorize`;
- a query whose tenant scope is disabled, or a tenant taken from anywhere but
  `data.Tenant(g)`;
- a `Grant` that can be produced without a policy returning nil;
- a field reaching a response that `Resource` does not list;
- a path from a request to the file system: a `Request` names no file, and the
  model paths are configuration;
- a prompt reaching a log line, an error or a column when `Config.KeepPrompts` is
  off;
- an image that makes the engine allocate or loop without bound -- every image is
  size-checked from its header before it is decoded;
- a heap or stack fault in the cgo boundary reachable from a request.

## What is not

- A vulnerability in the framework itself. Report that to the framework.
- A vulnerability in stable-diffusion.cpp or ggml themselves, or in a model file:
  report it to that project. A model file is code-adjacent input the application
  chose to load; this package digests what it loads so a run can be attributed,
  and does not vouch for the weights.
- A policy that is too permissive in an application that opened it. What this
  package ships denies everything; what an application opens is its own.
