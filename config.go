package diffusion

import (
	"fmt"
	"net/http"

	"github.com/arandu-io/framework/security"
)

// The defaults for the optional settings. They are constants rather than
// literals inside Config.withDefaults, so the value a reader finds here is the
// value the package uses.
const (
	// DefaultPrefix is where the routes are mounted when Config leaves Prefix
	// empty.
	DefaultPrefix = "/diffusion"
	// DefaultPageSize is how many records one page answers with when Config
	// leaves PageSize at zero.
	DefaultPageSize = 25
	// MaxPageSize is the ceiling PageSize is refused above. A page nobody
	// bounded is a page that reads the whole table on the day the table is
	// large.
	MaxPageSize = 200
)

// Config is what the application passes when it wires this package.
//
// A typed struct rather than a map: a misspelled key in a map is a setting that
// silently keeps its default, and the failure shows up as behaviour nobody
// asked for rather than as an error. Here a field that does not exist does not
// compile.
//
// It names no tenant, and that is deliberate. Every route sits behind
// RequireAuth, so the tenant of every statement is the one the Grant carries,
// from the session of whoever is asking; a tenant written here would be a
// second place for one to come from. The zero value is a valid configuration:
// it keeps and lists runs, and refuses everything until a policy opens an
// action.
type Config struct {
	// Prefix is the path the routes are mounted under. Empty means
	// DefaultPrefix.
	Prefix string

	// PageSize is how many records one page answers with. Zero means
	// DefaultPageSize, and anything above MaxPageSize is refused rather than
	// clamped: a number somebody wrote and did not get is worse than a number
	// somebody wrote and was told about.
	PageSize int

	// Engine is what generates. Nil means the module only keeps and lists runs
	// (Record, the two GET routes) and the generating route is not registered;
	// Generate answers ErrNoEngine. Pass the Engine that Load returned, or a Fake
	// in tests. The application owns it and closes it: the module only uses it.
	Engine Engine

	// KeepPrompts stores the text of each prompt with its run. It is off by
	// default: a prompt is often something a person wrote about themselves, and
	// the digest every run keeps is enough to tell that two runs asked for the
	// same thing. Turn it on when the product shows people their own history.
	KeepPrompts bool

	// Policy decides who may do what with a run.
	//
	// Nil is DiffusionPolicy, which denies everything -- so a wiring that says
	// nothing about rules gets no access at all, which is the state to start
	// from.
	//
	// It is here because an application installs this package with `go get` and
	// cannot edit the policy inside it. Without this field the only way to open
	// an action would be to fork the package, and a forked security boundary is
	// one that stops receiving the fixes made to the original. There is still
	// exactly one policy, consulted in exactly one place: what this changes is
	// who writes it, not how many of them there are.
	//
	// The rules go in the application, beside the rest of its authorization:
	//
	//	type DiffusionRules struct{}
	//
	//	func (DiffusionRules) Can(_ context.Context, s security.Subject, a security.Action, record diffusion.Diffusion) error {
	//		if record.ID != "" && record.TenantID != s.Tenant {
	//			return fmt.Errorf("the run belongs to another tenant")
	//		}
	//		if !s.Can(a) {
	//			return fmt.Errorf("the subject does not carry %s", a)
	//		}
	//		return nil
	//	}
	//
	//	func (DiffusionRules) Writes(a security.Action) bool { return diffusion.DiffusionPolicy{}.Writes(a) }
	//
	// The tenant comparison is the line to keep whatever else is written: every
	// rule below it holds across customers without it, as soon as two of them
	// have a record with the same identifier.
	Policy security.Policy[Diffusion]

	// Subject is who is acting on a request to this module's own routes.
	//
	// Nil reads the subject RequireAuth put on the request, which is the subject
	// of the realm the person signed in to. An application whose subject is more
	// than that session -- a member acting inside the organization they
	// selected, with the actions that membership carries and the organization as
	// the tenant -- passes the function its own handlers already build that
	// subject with, so the routes here are decided about by the same subject,
	// under the same tenant, as the rest of the application. The request has
	// already been admitted by RequireAuth when it is called.
	//
	// The tenant of the subject it answers is the tenant every statement is
	// scoped by, so it is the application's to establish from what it already
	// authorized -- the session, the membership -- and never a value the request
	// names. An error is the answer to the request.
	Subject func(*http.Request) (security.Subject, error)
}

// Validate reports what the configuration cannot be used with.
//
// It is called by New, so an application with a setting that cannot work fails
// where it is wired rather than on the first request that needed it.
func (c Config) Validate() error {
	if c.Prefix != "" && c.Prefix[0] != '/' {
		return fmt.Errorf("diffusion: Config.Prefix is %q and has to start with /", c.Prefix)
	}
	if c.Prefix != "" {
		if err := validateRoutePrefix(c.Prefix); err != nil {
			return err
		}
	}
	if c.PageSize < 0 || c.PageSize > MaxPageSize {
		return fmt.Errorf("diffusion: Config.PageSize is %d, and has to be between 0 and %d, where 0 means %d", c.PageSize, MaxPageSize, DefaultPageSize)
	}
	return nil
}

// validateRoutePrefix asks the standard library to parse the exact patterns
// the module will register. Its parser is not exported and reports invalid
// patterns by panic, so the throwaway mux turns that boot-time panic into the
// configuration error New promises.
func validateRoutePrefix(prefix string) (err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("diffusion: Config.Prefix %q cannot be registered as a route path", prefix)
		}
	}()

	handler := http.NotFoundHandler()
	mux := http.NewServeMux()
	mux.Handle(http.MethodGet+" "+prefix, handler)
	mux.Handle(http.MethodGet+" "+prefix+"/{id}", handler)
	mux.Handle(http.MethodPost+" "+prefix, handler)
	return nil
}

// withDefaults returns the configuration with the optional fields filled in.
//
// It runs after Validate and never before: filling a default in first would
// hide the value somebody actually wrote from the check that would have refused
// it.
func (c Config) withDefaults() Config {
	if c.Prefix == "" {
		c.Prefix = DefaultPrefix
	}
	if c.PageSize == 0 {
		c.PageSize = DefaultPageSize
	}
	return c
}
