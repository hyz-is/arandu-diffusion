// Package diffusion is an Arandu module that is also an image engine: it links
// stable-diffusion.cpp through cgo, so the model runs inside the process rather
// than behind an HTTP call, and it keeps what each run cost.
//
// There are two halves, and either works without the other.
//
// The engine is a library. Load turns a ModelSet -- a diffusion GGUF, its text
// encoder, its VAE and optional LoRAs -- into an Engine, and Engine.Generate
// does txt2img, img2img, inpainting under a mask that leaves the kept pixels
// identical, and reference-image editing, answering PNG bytes and the Metrics of
// the run. Fake is the same contract with no model, for an application's tests.
// The real engine exists only under the `sdcpp` build tag; without it the whole
// package compiles anywhere.
//
// The module is the Arandu half: one entity, Diffusion, which is one run kept;
// one policy that decides about it; one service that owns its Model-first data
// path; and the routes that reach them. The files are laid out by role rather
// than by layer, so the package reads top to bottom:
//
//	engine.go      -> the Engine contract, ModelSet, Request, Result, Metrics
//	engine_sdcpp.go-> the cgo engine (build tag sdcpp); engine_stub.go without it
//	fake.go        -> the scripted Engine for tests
//	imaging.go     -> decoding, masks, the composite that keeps masked pixels
//	modelfile.go   -> digests and the quantization read from a GGUF header
//	module.go      -> registration, routes, handlers and migrations
//	config.go      -> what the application passes in
//	model.go       -> the entity, and what it may answer with
//	policy.go      -> who may do what
//	service.go     -> the use cases, and Model access after authorization
//
// An application registers it explicitly. There is no service provider, no
// container and no discovery: the wiring is lines somebody wrote, and
// reading them is how they learn what the application is made of.
package diffusion

import (
	"context"
	"errors"
	stdhttp "net/http"
	"strconv"

	"github.com/arandu-io/framework/data"
	"github.com/arandu-io/framework/foundation"
	fhttp "github.com/arandu-io/framework/http"
	"github.com/arandu-io/framework/http/middleware"
	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/framework/validation"
	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
)

// Module is what the application registers.
//
// It implements foundation.Module, which is Name and Routes and nothing else --
// that pair is the whole public contract between a package and the framework.
//
// It also implements foundation.Migratable, because it owns a table. The other
// optional interfaces are declared beside Module in the framework and are opted
// into the same way, by implementing them: Bootable to prepare state at boot,
// Background to run a loop of its own, Schedulable to declare work for the
// scheduler, Health to report on the storage it depends on, Closable to give
// resources back at shutdown. It implements none of them: the engine it uses is
// the application's, and the application closes it.
type Module struct {
	cfg      Config
	svc      *DiffusionService
	sessions *security.SessionStore
}

// Compile-time proof that the module honors the contracts it claims.
var (
	_ foundation.Module     = (*Module)(nil)
	_ foundation.Migratable = (*Module)(nil)
)

// New returns the module, or the reason it cannot be built.
//
// The collaborators are parameters and not fields somebody fills in afterwards:
// a module that could be registered half-wired is a module whose first request
// is the thing that reports the missing half.
//
// It returns an error rather than panicking or carrying on, because everything
// it refuses is a wiring mistake, and a wiring mistake found at boot costs one
// restart. The same mistake found later is a request that reached a nil handle.
func New(cfg Config, db *data.DB, sessions *security.SessionStore) (*Module, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if db == nil {
		return nil, errors.New("diffusion: New needs a database handle: this package owns a table, and there is no in-memory mode that would let it start without one")
	}
	if sessions == nil {
		return nil, errors.New("diffusion: New needs a session store: every route is mounted behind RequireAuth, which reads it to admit a request and to put who is asking on it")
	}
	cfg = cfg.withDefaults()
	service := NewDiffusionService(db)
	service.engine = cfg.Engine
	service.keepPrompts = cfg.KeepPrompts
	if cfg.Policy != nil {
		// The one place the shipped policy is replaced, and it is a field of the
		// configuration the application already writes rather than a setter: a
		// service whose rules can be swapped after it has answered a request is
		// a service whose rules nobody can state.
		service.policy = cfg.Policy
	}
	return &Module{
		cfg:      cfg,
		svc:      service,
		sessions: sessions,
	}, nil
}

// Generate runs the engine for one subject and keeps the run. It is what an
// application calls from its own handler or job, where the images come from its
// own storage: the routes of this package only serve text-to-image.
//
// The policy is asked for DiffusionCreate before any compute is spent. See
// DiffusionService.Generate for what is kept.
func (m *Module) Generate(ctx context.Context, who security.Subject, req Request) (*Result, *Diffusion, error) {
	return m.svc.Generate(ctx, who, req)
}

// Record keeps a run the application made with the Engine itself, asking the
// same policy for the same action as Generate.
func (m *Module) Record(ctx context.Context, who security.Subject, run Run) (*Diffusion, error) {
	return m.svc.Record(ctx, who, run)
}

// Name is the module identifier: a lowercase slug, stable, no spaces.
//
// It is what `aru route:list` groups by and what the route names are prefixed with,
// so changing it changes addresses that other code has already written down.
func (m *Module) Name() string { return "diffusion" }

// Routes registers the module's routes under the configured prefix.
//
// They are named, so a URL is built from a name rather than written out a
// second time somewhere else -- two spellings of one address disagree, and the
// failure when they do is a link to a 404.
//
// Every route needs somebody signed in, so the guard is mounted once, on the
// router they are registered on, rather than route by route: r is replaced by
// the guarded group before the first registration, and no unguarded router is
// left in scope for the next route to be added to. RequireAuth sends a request
// with no session to the sign-in screen, and puts the subject of one that has a
// session on its context, where a handler reads it with ctx.User.
func (m *Module) Routes(r *fhttp.Router) {
	r = r.Group("", middleware.RequireAuth(m.sessions))

	r.Action(stdhttp.MethodGet, m.cfg.Prefix, m.index).Name("diffusion.index")
	r.Action(stdhttp.MethodGet, m.cfg.Prefix+"/{id}", m.show).Name("diffusion.show")
	// Generating is registered only when there is an engine to generate with: a
	// route that always answered "not configured" would be a route that exists
	// to be probed.
	if m.cfg.Engine != nil {
		r.Action(stdhttp.MethodPost, m.cfg.Prefix, m.store).Name("diffusion.store")
	}
}

// Handlers are thin on purpose: read the input, ask the service, answer. No
// rule, database handle or Model construction lives here. A handler that
// reached data directly would skip the service's policy boundary, and the
// layout makes that visible rather than relying on review.
//
// Who is asking comes from ctx.User, which reads the subject RequireAuth put on
// the request. No handler loads the session again or takes an identity from the
// input. The second value ctx.User answers is not consulted: behind the guard
// there is always a subject, and were there none, the zero subject it answers
// would be refused by security.Authorize before any policy ran.
//
// An error from the service is returned as it is, and the router answers it:
// validation.Errors goes back where the request came from with the messages in
// the flash, ErrNotFound is 404 because it wraps model.ErrModelNotFound, and a
// policy refusal is 403. The answer is the status and its standard sentence,
// never the error's own text -- telling a client why a policy said no tells it
// what exists, one request at a time, and the reason belongs in the log. Any
// other error is one nobody expected, and it reaches the error page rather than
// a 200 with an empty body.

// acting is the subject a request is decided about: the one RequireAuth admitted,
// or the one the application's own Config.Subject builds from the request.
func (m *Module) acting(r *stdhttp.Request, admitted security.Subject) (security.Subject, error) {
	if m.cfg.Subject != nil {
		return m.cfg.Subject(r)
	}
	return admitted, nil
}

// index answers a page of runs.
func (m *Module) index(ctx *fhttp.Context) error {
	query := data.Query{
		Sort:   ctx.Query("sort"),
		Cursor: ctx.Query("cursor"),
		Limit:  m.cfg.PageSize,
	}

	admitted, _ := ctx.User()
	who, err := m.acting(ctx.Request, admitted)
	if err != nil {
		return err
	}
	records, err := m.svc.List(ctx.Ctx(), who, query)
	if err != nil {
		return err
	}

	// A full page is the only one that can have a successor. A short page is
	// the last one, and offering a cursor for it would be offering a next page
	// that comes back empty.
	cursor := ""
	if len(records) == m.cfg.PageSize {
		cursor = records[len(records)-1].ID
	}
	return ctx.JSON(stdhttp.StatusOK, collectionFromPointers(records, cursor))
}

// show answers one run.
func (m *Module) show(ctx *fhttp.Context) error {
	admitted, _ := ctx.User()
	who, err := m.acting(ctx.Request, admitted)
	if err != nil {
		return err
	}
	record, err := m.svc.Find(ctx.Ctx(), who, ctx.Param("id"))
	if err != nil {
		return err
	}
	return ctx.JSON(stdhttp.StatusOK, resourceFromPointer(record))
}

// store generates one picture from a prompt and answers it as image/png, with
// the run's identifier and seed in headers so the client can look the run up
// and repeat it.
//
// It takes text-to-image only. An image arriving over this route would be a
// second place a file enters the application, beside the storage the
// application already guards; img2img, inpainting and editing are Module.Generate,
// called from the handler that read the image from there.
//
// The call holds the connection for as long as the model takes, which on a CPU
// is minutes. An application that serves people puts the run on its queue and
// calls Module.Generate from the worker; this route is for an operator and for
// development.
func (m *Module) store(ctx *fhttp.Context) error {
	req, err := generateRequest(ctx)
	if err != nil {
		return err
	}
	admitted, _ := ctx.User()
	who, err := m.acting(ctx.Request, admitted)
	if err != nil {
		return err
	}
	result, record, err := m.svc.Generate(ctx.Ctx(), who, req)
	if err != nil {
		return err
	}
	h := ctx.Response.Header()
	h.Set("Content-Type", "image/png")
	h.Set("Cache-Control", "private, no-store")
	h.Set("X-Content-Type-Options", "nosniff")
	h.Set("X-Diffusion-Run", record.ID)
	h.Set("X-Diffusion-Seed", strconv.FormatInt(result.Seed, 10))
	ctx.Response.WriteHeader(stdhttp.StatusCreated)
	_, err = ctx.Response.Write(result.PNG)
	return err
}

// generateRequest reads the text-to-image fields. A number that is not a number
// is a rejected input, named by field, and never a default: a width the client
// wrote and did not get is worse than one it was told about.
func generateRequest(ctx *fhttp.Context) (Request, error) {
	errs := validation.Errors{}
	whole := func(name string) int {
		v := ctx.Input(name)
		if v == "" {
			return 0
		}
		n, err := strconv.Atoi(v)
		if err != nil {
			errs.Add(name, "must be a whole number")
		}
		return n
	}
	decimal := func(name string) float64 {
		v := ctx.Input(name)
		if v == "" {
			return 0
		}
		n, err := strconv.ParseFloat(v, 64)
		if err != nil {
			errs.Add(name, "must be a number")
		}
		return n
	}
	req := Request{
		Prompt:         ctx.Input("prompt"),
		NegativePrompt: ctx.Input("negative_prompt"),
		Width:          whole("width"),
		Height:         whole("height"),
		Steps:          whole("steps"),
		CFG:            decimal("cfg"),
		Sampler:        ctx.Input("sampler"),
		Scheduler:      ctx.Input("scheduler"),
		// An absent seed is a random one, which the response reports.
		Seed: -1,
	}
	if v := ctx.Input("seed"); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil {
			errs.Add("seed", "must be a whole number")
		}
		req.Seed = n
	}
	if errs.Any() {
		return Request{}, errs
	}
	return req, nil
}

// Migrations declares the schema this module owns.
//
// They are returned in the order their names sort in, which is the order they
// apply in: the name carries the order, and nothing else decides it.
func (m *Module) Migrations() []foundation.Migration {
	return []foundation.Migration{createDiffusions{}}
}

// The migration is reversible, and the assertion is here rather than discovered
// at rollback: the migrator tests for Down with a type assertion, so a Down
// with the wrong signature would leave a rollback that silently does nothing.
var _ migrations.ReversibleMigration = createDiffusions{}

// createDiffusions is the table this module owns, and the index its listing
// reads by.
type createDiffusions struct{ migrations.BaseMigration }

// GetName is the migration's identity, and it carries the order. It is fixed
// once the package is published: changing what an applied name means leaves the
// change missing everywhere it already ran, and nothing says so.
func (createDiffusions) GetName() string { return "20261120_0001_create_diffusions" }

// Up creates the table and the index the keyset pagination scans.
//
// The Blueprint spells each column for the engine the migration is running on,
// which is what lets one application develop on a file and deploy on Postgres
// without a second schema. That used to be written out as SQL here, with a
// comment explaining that an identifier column is VARCHAR rather than TEXT
// because it takes part in a key and MySQL refuses TEXT in one without a prefix
// length -- which is the grammar's job, done by hand in every migration that
// remembered to.
//
// The timestamp has no database default: the value comes from Go.
func (createDiffusions) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Create(ctx, "diffusions", func(table *schema.Blueprint) {
		table.String("id").Primary()
		table.String("tenant_id")
		// Who asked, what for, how it ended.
		table.String("user_id")
		table.String("mode")
		table.String("status")
		table.String("error", 400)
		// The prompt is stored only under Config.KeepPrompts; the digest always.
		table.Text("prompt")
		table.String("prompt_sha256")
		// Which weights ran. These are columns and not a note because the
		// question a run answers is what a model costs, and that is a
		// comparison across them with everything else held equal.
		table.String("model_name")
		table.String("model_sha256")
		table.String("quantization")
		table.String("backend")
		table.Integer("threads")
		// The knobs, the seed being the one the engine used.
		table.Integer("width")
		table.Integer("height")
		table.Integer("steps")
		table.BigInteger("seed")
		table.String("sampler")
		table.Float("cfg")
		// What it cost.
		table.BigInteger("duration_ms")
		table.BigInteger("sampling_ms")
		table.BigInteger("peak_rss_bytes")
		table.Timestamp("created_at")

		// The index matches the ORDER BY of the listing, tenant first. Without
		// it every page is a scan of every customer's rows.
		table.Index([]string{"tenant_id", "created_at", "id"}, "diffusions_tenant_created_idx")

		// The question asked of this table is almost always "how did the cost
		// move as the model, the quantization or the backend changed". Without
		// this index that is a scan.
		table.Index([]string{"tenant_id", "model_sha256", "created_at"}, "diffusions_tenant_model_idx")
	})
}

// Down drops the table, which takes its index with it.
func (createDiffusions) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().DropIfExists(ctx, "diffusions")
}
