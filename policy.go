package diffusion

import (
	"context"
	"fmt"

	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/hesape/auth"
)

// The actions of Diffusion. Constants rather than strings at the call site: a
// typo in an action name would silently authorize nothing, or worse, everything.
//
// They carry the entity in the name because an application registers many
// packages, and the name of an action shows up in logs and in audit trails
// where "view" on its own says nothing about what was viewed.
const (
	// DiffusionView is reading one run.
	DiffusionView security.Action = "diffusion.view"
	// DiffusionList is paging through the runs.
	DiffusionList security.Action = "diffusion.list"
	// DiffusionCreate is running the engine and keeping the run, or keeping a run
	// the application made itself. It is the action that spends compute.
	DiffusionCreate security.Action = "diffusion.create"
)

// DiffusionPolicy is the only authority over who does what with a Diffusion.
//
// IT DENIES EVERYTHING, and that is the state to start from rather than a
// placeholder to delete. A policy shipped with a branch that allows every
// action is a hole in every application that installs the package, and the hole
// looks like working code until somebody reads it.
//
// There is deliberately no such branch to remove. Open one action at a time,
// inside the custom block below, saying who may take it and on which record --
// what is not written there stays closed, including every action added later.
type DiffusionPolicy struct{}

// Compile-time proof that the policy answers about this entity and no other. A
// policy that drifted onto another type would leave this one unguarded while
// the Model path still compiled.
var _ security.Policy[Diffusion] = DiffusionPolicy{}

// Compile-time proof that the policy says which of its actions write.
var _ auth.Writes = DiffusionPolicy{}

// Writes reports whether a Grant issued for the action may write. DiffusionView
// and DiffusionList read, so the Grant Authorize issues for either is refused by
// every write -- the database, the cache, the outbox -- with auth.ErrReadGrant.
//
// The set is closed and written out: an action added later reads until it is
// named here, so a Grant issued for it cannot write by accident. An application
// that passes its own policy in Config.Policy answers Writes by calling this
// method, so both policies agree on which actions write.
func (DiffusionPolicy) Writes(a security.Action) bool { return a == DiffusionCreate }

// Can decides whether the subject may perform the action on the record.
//
// It is the only place that decides. The service reaches the Model only after
// Authorize turns this method's nil result into a Grant.
func (DiffusionPolicy) Can(ctx context.Context, s security.Subject, a security.Action, record Diffusion) error {
	// Tenant isolation comes first and applies to every action. Without it every
	// check below would be pointless in a multi-tenant system: a rule that
	// allows an owner to read their own record would allow it across customers
	// as soon as two of them have a record with the same identifier.
	//
	// The empty id is the candidate that has not been stored yet, which belongs
	// to nobody until it is written with the tenant off the Grant.
	if record.ID != "" && record.TenantID != s.Tenant {
		return fmt.Errorf("diffusion belongs to another tenant")
	}

	// arandu:begin custom
	// The rules of this package go here, one action at a time. A rule that
	// depends on the record and not only on the role is written the same way:
	//
	//	if a == DiffusionView && (s.ID == record.ID || s.HasRole("admin")) {
	//		return nil
	//	}
	//
	// The routes of this package sit behind RequireAuth, so none of them asks
	// about a guest. An application that calls the service from a public route
	// of its own may, with security.Guest: a reader it declared anonymous on
	// purpose, and the only subject that arrives without an id. Answer it
	// explicitly or it falls through to the refusal below, which is the safe
	// direction:
	//
	//	if a == DiffusionView && s.IsGuest() && record.Published {
	//		return nil
	//	}
	// arandu:end custom

	return fmt.Errorf("no rule allows %s on diffusion", a)
}
