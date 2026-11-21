---
name: diffusion-policy
description: Open an action of the policy, add an authorized use case to the service, or understand a 403 from this package. Use when the request is to "let admins generate", "open diffusion.create", "who may list runs", "403 from /diffusion", "add a service method", "authorize before the model", "the tenant comes from where", or when a change touches policy.go or service.go.
license: MIT
---

# Who may do what, and in which order

## The three actions

| action | what it is | writes |
| --- | --- | --- |
| `DiffusionView` | reading one run | no |
| `DiffusionList` | paging through the runs | no |
| `DiffusionCreate` | running the engine and keeping the run, or keeping a run made elsewhere | yes |

`DiffusionCreate` is the action that spends compute, and it is asked **before**
the engine runs. `DiffusionPolicy.Writes` says so (`auth.Writes`), so the Grant
issued for a view or a list refuses every write with `auth.ErrReadGrant`.

## The policy denies everything

`DiffusionPolicy` has no branch that allows an action, and that is the state the
package ships in. An application installs it with `go get` and cannot edit the
file, so the rules go in `Config.Policy`, in the application:

```go
type DiffusionRules struct{}

func (DiffusionRules) Can(_ context.Context, s security.Subject, a security.Action, record diffusion.Diffusion) error {
	if record.ID != "" && record.TenantID != s.Tenant {
		return fmt.Errorf("the run belongs to another tenant")
	}
	if !s.Can(a) { // Actions filled by arandu-permission
		return fmt.Errorf("the subject does not carry %s", a)
	}
	return nil
}

func (DiffusionRules) Writes(a security.Action) bool { return diffusion.DiffusionPolicy{}.Writes(a) }
```

The tenant comparison is the line to keep. Permissions and roles come from
`hyz-is/arandu-permission`; this package never grows a role table of its own.

## The order in a service method

`validate -> security.Authorize -> Grant -> engine -> Diffusions(s.db).New() -> Save(ctx, g)`

- A request is validated before it is authorized, so a malformed one costs nothing.
- `security.Authorize` produces the Grant; the candidate it sees carries the mode
  and the person, so a rule can be about *this* run.
- The row is built with `Diffusions(s.db).New()` and its tenant is
  `data.Tenant(g)`. Never a tenant from the request, a header or a config field.
- A failed or cancelled run is **kept** (the same Grant, on `context.WithoutCancel`):
  it cost the same compute, and a comparison of models is of all their runs.
- `tests/Unit/audit_test.go` reads every exported service method for the order;
  `save` counts as the data boundary because it only takes a Grant.

## A 403 from the routes

Nobody opened the action, or the subject's `Actions` do not carry it. The
response carries no reason; the reason is the error `Can` returned, in the log.
