package diffusion

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/arandu-io/framework/data"
	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/framework/validation"
)

// ErrNoEngine is Generate on a module configured without an Engine.
var ErrNoEngine = errors.New("diffusion: no engine is configured; pass Config.Engine")

// Pagination bounds for List. A request that asks for everything gets the
// maximum, never everything: an unbounded query is how one page load takes a
// production database down.
const (
	defaultLimit = 50
	maxLimit     = 200
)

// sortableDiffusion is the ordering allowlist. A column name taken directly
// from a request would turn ordering into an injection surface.
var sortableDiffusion = map[string]string{
	"":            "created_at",
	"created_at":  "created_at",
	"duration_ms": "duration_ms",
}

// DiffusionService orchestrates the use cases of this package: it validates the
// request, asks the policy for a Grant, runs the engine, and persists the run
// with that Grant.
//
// It receives its collaborators through the constructor. There is no container
// and no resolution by reflection: what this service is made of is written at
// the one place that builds it, and reading that place is how somebody learns
// what the package touches.
//
// Everything a handler is allowed to do goes through here. The service is the
// only owner of the database handle, so the request layer cannot reach a Model
// before the policy has answered.
type DiffusionService struct {
	db          *data.DB
	policy      security.Policy[Diffusion]
	engine      Engine
	keepPrompts bool
}

// NewDiffusionService wires the service over the application's database handle.
// It installs DiffusionPolicy, which denies everything, and no engine.
func NewDiffusionService(db *data.DB) *DiffusionService {
	return &DiffusionService{db: db, policy: DiffusionPolicy{}}
}

// Run is what happened: the request, and either its result or the error that
// ended it. It is the input of Record, for an application that runs the engine
// itself -- in a queue worker, with its own timeout -- and keeps the run through
// this package.
type Run struct {
	// Request is what was asked. It is read for the knobs and the prompt; its
	// images are never stored.
	Request Request
	// Result is the finished run, nil when Err is set.
	Result *Result
	// Err is why the run did not finish, nil when it did.
	Err error
}

// Validate reports what cannot be kept.
func (r Run) Validate() validation.Errors {
	e := validation.Errors{}
	if r.Result == nil && r.Err == nil {
		e.Add("run", "a run has a result or an error")
	}
	if r.Result != nil && r.Err != nil {
		e.Add("run", "a run cannot have both a result and an error")
	}
	return e
}

// Compile-time proof that the run honors the validation contract.
var _ validation.Validatable = Run{}

// Generate is the whole path in one function: validate, authorize, run the
// engine with the Grant's decision already made, keep the run.
//
// The policy is asked before the engine is, so a subject who may not generate
// spends nothing, and the candidate it sees carries the mode and the person: a
// rule about what may be generated is a rule about this run. The run is kept
// whether it succeeded, failed or was cancelled -- a failed run costs the same
// compute as a good one, and the comparison between models is of all of them --
// and it is kept on a context that outlives the caller's, since the usual reason
// a run ends early is that the caller's context did.
//
// It returns the result and the stored run. When the engine fails it returns
// the engine's error, joined with the storage error if keeping the failure
// failed too.
func (s *DiffusionService) Generate(ctx context.Context, actor security.Subject, req Request) (*Result, *Diffusion, error) {
	if s.engine == nil {
		return nil, nil, ErrNoEngine
	}
	if err := req.Validate(); err != nil {
		return nil, nil, asValidation(err)
	}

	g, err := security.Authorize(ctx, s.policy, actor, DiffusionCreate, Diffusion{UserID: actor.ID, Mode: string(req.Mode())})
	if err != nil {
		return nil, nil, err
	}

	result, runErr := s.engine.Generate(ctx, req)
	row, saveErr := s.save(context.WithoutCancel(ctx), g, actor, Run{Request: req, Result: result, Err: runErr})
	if runErr != nil {
		return nil, nil, errors.Join(asValidation(runErr), saveErr)
	}
	if saveErr != nil {
		// The picture exists and costs minutes to make again, so it is returned
		// beside the error that says the run was not kept.
		return result, nil, saveErr
	}
	return result, row, nil
}

// Record keeps a run the application made itself. It asks the same policy for
// the same action as Generate, because keeping a run is claiming one was made.
func (s *DiffusionService) Record(ctx context.Context, actor security.Subject, run Run) (*Diffusion, error) {
	if errs := run.Validate(); errs.Any() {
		return nil, errs
	}
	g, err := security.Authorize(ctx, s.policy, actor, DiffusionCreate, Diffusion{UserID: actor.ID, Mode: string(run.Request.Mode())})
	if err != nil {
		return nil, err
	}
	return s.save(ctx, g, actor, run)
}

// save writes one run with a Grant that was issued for DiffusionCreate.
func (s *DiffusionService) save(ctx context.Context, g security.Grant, actor security.Subject, run Run) (*Diffusion, error) {
	id, err := data.NewID()
	if err != nil {
		return nil, err
	}
	// The row is built by the query so it is wired to its table, and its fields
	// are assigned one by one: copying a struct literal over it would replace the
	// embedded model with the unwired one.
	row, err := Diffusions(s.db).New()
	if err != nil {
		return nil, err
	}
	row.ID = id
	row.TenantID = data.Tenant(g)
	row.UserID = actor.ID
	fillRun(row, run, s.engine, s.keepPrompts)
	if _, err := row.Save(ctx, g); err != nil {
		return nil, err
	}
	return row, nil
}

// fillRun copies what a run says about itself into a row. It is a plain function
// of its inputs, so what is stored for a failure and for a success can be read
// -- and tested -- in one place.
func fillRun(row *Diffusion, run Run, engine Engine, keepPrompts bool) {
	req := run.Request
	row.Mode = string(req.Mode())
	sum := sha256.Sum256([]byte(req.Prompt))
	row.PromptSHA256 = hex.EncodeToString(sum[:])
	if keepPrompts {
		row.Prompt = req.Prompt
	}
	row.Sampler = req.Sampler
	row.CFG = req.CFG
	row.Width, row.Height, row.Seed, row.Steps = req.Width, req.Height, req.Seed, req.Steps

	var info Info
	if engine != nil {
		info = engine.Info()
	}
	models := info.Models
	row.ModelName, row.Backend, row.Threads = info.Name, info.Backend, info.Threads

	switch {
	case run.Err != nil:
		row.Status = StatusFailed
		if errors.Is(run.Err, context.Canceled) || errors.Is(run.Err, context.DeadlineExceeded) {
			row.Status = StatusCancelled
		}
		row.Error = shorten(run.Err.Error(), req.Prompt)
	case run.Result != nil:
		res := run.Result
		row.Status = StatusOK
		row.Mode = string(res.Mode)
		row.Width, row.Height, row.Seed, row.Steps = res.Width, res.Height, res.Seed, res.Metrics.Steps
		row.DurationMS = res.Metrics.Total.Milliseconds()
		row.SamplingMS = res.Metrics.Sampling.Milliseconds()
		row.PeakRSSBytes = int64(min(res.Metrics.PeakRSSBytes, 1<<62))
		if res.Metrics.Backend != "" {
			row.Backend, row.Threads = res.Metrics.Backend, res.Metrics.Threads
		}
		if len(res.Metrics.Models) > 0 {
			models = res.Metrics.Models
		}
	}
	for _, m := range models {
		if m.Role == "checkpoint" || m.Role == "diffusion" {
			row.ModelSHA256, row.Quantization = m.SHA256, m.Quantization
			if row.ModelName == "" {
				row.ModelName = m.Name
			}
			break
		}
	}
}

// shorten makes an error fit a column and keeps the prompt out of it: a library
// message that quotes what the person wrote is not a message to store.
func shorten(msg, prompt string) string {
	if prompt != "" {
		msg = strings.ReplaceAll(msg, prompt, "[prompt]")
	}
	if len(msg) <= ErrorMaxLen {
		return msg
	}
	cut := msg[:ErrorMaxLen]
	for !utf8.ValidString(cut) && len(cut) > 0 {
		cut = cut[:len(cut)-1]
	}
	return cut
}

// asValidation turns a request the engine refused into the validation error the
// router answers with 422, and leaves every other error as it is.
func asValidation(err error) error {
	if err == nil || !errors.Is(err, ErrInvalidRequest) {
		return err
	}
	e := validation.Errors{}
	e.Add("request", strings.TrimPrefix(err.Error(), ErrInvalidRequest.Error()+": "))
	return e
}

// Find returns one record, and asks the policy twice.
//
// The first call is on the empty candidate, because there is no way to read the
// record without a Grant and no way to hold a Grant without a decision. What it
// decides is whether this subject may view records of this kind at all.
//
// The second call is on the record that came back, and it is the one a rule
// about the record itself depends on: the first call saw an empty value, so
// anything the policy says about who owns the row, or about a row that is not
// published yet, never ran. Without it a policy can be written that looks
// correct, reads correctly, and is never consulted about the thing it protects.
//
// The read itself is already scoped by data.Tenant, so the second call is not
// what keeps customers apart. It is what keeps the policy honest.
func (s *DiffusionService) Find(ctx context.Context, actor security.Subject, id string) (*Diffusion, error) {
	g, err := security.Authorize(ctx, s.policy, actor, DiffusionView, Diffusion{})
	if err != nil {
		return nil, err
	}

	record, err := Diffusions(s.db).WhereKey(id).First(ctx, g)
	if err != nil {
		return nil, err
	}
	if record == nil {
		return nil, ErrNotFound
	}

	if _, err := security.Authorize(ctx, s.policy, actor, DiffusionView, *record); err != nil {
		return nil, err
	}
	return record, nil
}

// List returns a page of records.
//
// It authorizes once, on the empty candidate, and the tenant filter in the
// statement is what bounds the rows. A policy call per row would be one call
// per record on a page and would still not narrow the query -- a listing that
// has to read a customer's rows in order to decide it may not read them has
// already read them.
//
// A rule that hides individual records from a listing belongs in the statement,
// as a predicate, and the action here is what decides whether the listing may
// run at all.
func (s *DiffusionService) List(ctx context.Context, actor security.Subject, q data.Query) ([]*Diffusion, error) {
	g, err := security.Authorize(ctx, s.policy, actor, DiffusionList, Diffusion{})
	if err != nil {
		return nil, err
	}

	column, ok := sortableDiffusion[q.Sort]
	if !ok {
		return nil, fmt.Errorf("diffusion: sort field not allowed: %q", q.Sort)
	}

	limit := q.Limit
	switch {
	case limit <= 0:
		limit = defaultLimit
	case limit > maxLimit:
		limit = maxLimit
	}

	page := Diffusions(s.db)
	if q.Cursor != "" {
		anchor, err := Diffusions(s.db).WhereKey(q.Cursor).Value(ctx, g, column)
		if err != nil {
			return nil, err
		}
		if anchor == nil {
			return nil, nil
		}
		page = page.Where(func(after *DiffusionQuery) {
			after.Where(column, ">", anchor).
				OrWhere(func(equal *DiffusionQuery) {
					equal.Where(column, "=", anchor).Where("id", ">", q.Cursor)
				})
		})
	}

	return page.OrderBy(column).OrderBy("id").Limit(limit).Get(ctx, g)
}
