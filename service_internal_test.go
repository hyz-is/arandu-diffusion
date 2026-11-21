package diffusion

import (
	"context"
	"database/sql"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/arandu-io/framework/data"
	"github.com/arandu-io/framework/security"
	"github.com/arandu-io/hesape/database"

	_ "modernc.org/sqlite"
)

// The tests in this file run against a database, and they are internal for the
// reason the same tests are in the other Arandu modules: the shipped policy
// refuses everything, so reaching the service at all means holding a policy that
// answers yes, and the field that holds one is unexported.
//
// The engine is the Fake. What is under test is what the module keeps about a
// run and for whom, not the picture.

// openPolicy allows every action and says nothing about tenants, deliberately:
// the tenant filter under test is the one in the statement, and a policy that
// also refused a foreign record would answer for it.
type openPolicy struct{}

var _ security.Policy[Diffusion] = openPolicy{}

func (openPolicy) Can(context.Context, security.Subject, security.Action, Diffusion) error {
	return nil
}

func (openPolicy) Writes(a security.Action) bool { return DiffusionPolicy{}.Writes(a) }

func openDatabase(t *testing.T, name string) *data.DB {
	t.Helper()

	raw, err := sql.Open("sqlite", "file:"+name+"?mode=memory&cache=shared")
	if err != nil {
		t.Fatalf("opening the database: %v", err)
	}
	t.Cleanup(func() { _ = raw.Close() })
	raw.SetMaxOpenConns(1)

	migrator := database.ForMigrations(database.NewConnection(raw, "", "", map[string]any{
		"driver": string(database.DialectSQLite),
		"name":   name,
	}))
	// The module's own declaration is applied, so the schema this suite exercises
	// and the schema an installer gets cannot drift apart.
	for _, migration := range (&Module{}).Migrations() {
		if err := migration.Up(context.Background(), migrator); err != nil {
			t.Fatalf("applying %s: %v", migration.GetName(), err)
		}
	}
	return data.Wrap(raw, data.DialectSQLite)
}

func openService(t *testing.T, name string, engine Engine, keepPrompts bool) *DiffusionService {
	t.Helper()
	s := NewDiffusionService(openDatabase(t, name))
	s.policy = openPolicy{}
	s.engine = engine
	s.keepPrompts = keepPrompts
	return s
}

func memberOf(tenant, id string) security.Subject {
	return security.Subject{ID: id, Tenant: tenant, Verified: true}
}

func TestGenerateKeepsTheRunWithWhoWhatAndHowLong(t *testing.T) {
	t.Parallel()

	fake := NewFake()
	s := openService(t, "keeps", fake, false)
	ctx := context.Background()

	result, row, err := s.Generate(ctx, memberOf("acme", "ana"), Request{
		Prompt: "a red apple on a table", Width: 64, Height: 64, Seed: 9, Steps: 3, CFG: 6.5, Sampler: "euler",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(result.PNG) == 0 || row == nil {
		t.Fatal("no picture or no run")
	}

	found, err := s.Find(ctx, memberOf("acme", "ana"), row.ID)
	if err != nil {
		t.Fatalf("Find: %v", err)
	}
	if found.TenantID != "acme" || found.UserID != "ana" {
		t.Fatalf("the run belongs to %q/%q, want acme/ana", found.TenantID, found.UserID)
	}
	if found.Status != StatusOK || found.Error != "" || found.Mode != "txt2img" {
		t.Fatalf("status=%q error=%q mode=%q", found.Status, found.Error, found.Mode)
	}
	if found.Width != 64 || found.Height != 64 || found.Steps != 3 || found.Seed != 9 || found.Sampler != "euler" || found.CFG != 6.5 {
		t.Fatalf("the knobs were not kept: %+v", found)
	}
	if found.ModelSHA256 != fakeDigest || found.Quantization != "Q8_0" || found.ModelName != "fake" || found.Backend != "fake" {
		t.Fatalf("the model is not identified: %q %q %q %q", found.ModelSHA256, found.Quantization, found.ModelName, found.Backend)
	}
	if found.PromptSHA256 == "" || len(found.PromptSHA256) != 64 {
		t.Fatalf("prompt digest = %q", found.PromptSHA256)
	}
	if found.Prompt != "" {
		t.Fatalf("the prompt %q was stored without Config.KeepPrompts", found.Prompt)
	}
	if found.CreatedAt.IsZero() || time.Since(found.CreatedAt) > time.Minute {
		t.Fatalf("created_at = %v", found.CreatedAt)
	}
}

func TestThePromptIsKeptOnlyWhenTheApplicationOptedIn(t *testing.T) {
	t.Parallel()

	s := openService(t, "prompts", NewFake(), true)
	_, row, err := s.Generate(context.Background(), memberOf("acme", "ana"), Request{Prompt: "my own words", Width: 64, Height: 64, Steps: 1})
	if err != nil {
		t.Fatal(err)
	}
	if row.Prompt != "my own words" {
		t.Fatalf("prompt = %q, want it kept", row.Prompt)
	}
}

func TestAFailedRunIsKeptWithoutThePrompt(t *testing.T) {
	t.Parallel()

	prompt := "a secret about my neighbour"
	boom := errors.New("library said: could not encode '" + prompt + "' ")
	s := openService(t, "failed", NewFake(Answer{Err: boom}), false)
	ctx := context.Background()

	if _, _, err := s.Generate(ctx, memberOf("acme", "ana"), Request{Prompt: prompt}); !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the engine's", err)
	}
	rows, err := s.List(ctx, memberOf("acme", "ana"), data.Query{})
	if err != nil || len(rows) != 1 {
		t.Fatalf("List = %d rows, %v; a failed run costs compute and is kept", len(rows), err)
	}
	if rows[0].Status != StatusFailed || !strings.Contains(rows[0].Error, "could not encode") {
		t.Fatalf("status=%q error=%q", rows[0].Status, rows[0].Error)
	}
	if strings.Contains(rows[0].Error, prompt) || rows[0].Prompt != "" {
		t.Fatalf("the prompt reached the row: error=%q prompt=%q", rows[0].Error, rows[0].Prompt)
	}
}

func TestACancelledRunIsKeptEvenThoughItsContextIsDead(t *testing.T) {
	t.Parallel()

	fake := NewFake()
	fake.Delay = 10 * time.Millisecond
	s := openService(t, "cancelled", fake, false)
	ctx, cancel := context.WithCancel(context.Background())
	_, _, err := s.Generate(ctx, memberOf("acme", "ana"), Request{
		Prompt: "x", Width: 64, Height: 64, Steps: 50,
		OnProgress: func(Progress) { cancel() },
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	rows, err := s.List(context.Background(), memberOf("acme", "ana"), data.Query{})
	if err != nil || len(rows) != 1 || rows[0].Status != StatusCancelled {
		t.Fatalf("rows = %d, %v; want one run, cancelled", len(rows), err)
	}
}

func TestARefusedSubjectSpendsNoCompute(t *testing.T) {
	t.Parallel()

	fake := NewFake()
	s := NewDiffusionService(openDatabase(t, "refused")) // the shipped policy: denies all
	s.engine = fake

	_, _, err := s.Generate(context.Background(), memberOf("acme", "ana"), Request{Prompt: "x"})
	if !errors.Is(err, security.ErrForbidden) {
		t.Fatalf("err = %v, want ErrForbidden", err)
	}
	if n := len(fake.Requests()); n != 0 {
		t.Fatalf("the engine was asked %d times by a subject the policy refused", n)
	}
	// A request the policy would also refuse is refused as invalid first, with
	// nothing spent either way.
	if _, _, err := s.Generate(context.Background(), memberOf("acme", "ana"), Request{}); err == nil {
		t.Fatal("an empty request was accepted")
	}
	if len(fake.Requests()) != 0 {
		t.Fatal("the engine saw an invalid request")
	}
}

func TestWithoutAnEngineGenerateSaysSo(t *testing.T) {
	t.Parallel()

	s := openService(t, "noengine", nil, false)
	if _, _, err := s.Generate(context.Background(), memberOf("acme", "ana"), Request{Prompt: "x"}); !errors.Is(err, ErrNoEngine) {
		t.Fatalf("err = %v, want ErrNoEngine", err)
	}
}

func TestOneTenantNeverReadsAnothersRuns(t *testing.T) {
	t.Parallel()

	s := openService(t, "tenants", NewFake(), false)
	ctx := context.Background()
	_, mine, err := s.Generate(ctx, memberOf("acme", "ana"), Request{Prompt: "x", Width: 64, Height: 64, Steps: 1})
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := s.Generate(ctx, memberOf("globex", "bo"), Request{Prompt: "y", Width: 64, Height: 64, Steps: 1}); err != nil {
		t.Fatal(err)
	}

	if rows, err := s.List(ctx, memberOf("acme", "ana"), data.Query{}); err != nil || len(rows) != 1 || rows[0].TenantID != "acme" {
		t.Fatalf("acme listed %d rows (%v), want its own one", len(rows), err)
	}
	// The row exists, in another tenant: it answers not found, the same as a row
	// that never existed.
	if _, err := s.Find(ctx, memberOf("globex", "bo"), mine.ID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("globex read acme's run: err = %v, want ErrNotFound", err)
	}
}

func TestRecordKeepsARunTheApplicationMadeItself(t *testing.T) {
	t.Parallel()

	s := openService(t, "record", nil, false)
	ctx := context.Background()
	fake := NewFake()
	req := Request{Prompt: "queued run", Width: 64, Height: 64, Seed: 4, Steps: 2}
	res, err := fake.Generate(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	row, err := s.Record(ctx, memberOf("acme", "ana"), Run{Request: req, Result: res})
	if err != nil {
		t.Fatalf("Record: %v", err)
	}
	if row.Status != StatusOK || row.Seed != 4 || row.ModelSHA256 != fakeDigest || row.Backend != "fake" {
		t.Fatalf("recorded %+v", row)
	}
	if _, err := s.Record(ctx, memberOf("acme", "ana"), Run{Request: req}); err == nil {
		t.Fatal("a run with neither a result nor an error was kept")
	}
}

func TestTheListPagesByCursor(t *testing.T) {
	t.Parallel()

	s := openService(t, "paging", NewFake(), false)
	ctx := context.Background()
	for range 5 {
		if _, _, err := s.Generate(ctx, memberOf("acme", "ana"), Request{Prompt: "x", Width: 64, Height: 64, Steps: 1}); err != nil {
			t.Fatal(err)
		}
	}
	seen := map[string]bool{}
	cursor := ""
	for pages := 0; pages < 10; pages++ {
		rows, err := s.List(ctx, memberOf("acme", "ana"), data.Query{Cursor: cursor, Limit: 2})
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range rows {
			if seen[r.ID] {
				t.Fatalf("run %s listed twice", r.ID)
			}
			seen[r.ID] = true
		}
		if len(rows) < 2 {
			break
		}
		cursor = rows[len(rows)-1].ID
	}
	if len(seen) != 5 {
		t.Fatalf("paging reached %d of 5 runs", len(seen))
	}
	if _, err := s.List(ctx, memberOf("acme", "ana"), data.Query{Sort: "prompt; drop table"}); err == nil {
		t.Fatal("a sort outside the allowlist was accepted")
	}
}

func TestAnErrorThatQuotesThePromptIsShortened(t *testing.T) {
	t.Parallel()

	long := strings.Repeat("é", ErrorMaxLen)
	if got := shorten(long, ""); len(got) > ErrorMaxLen || !strings.HasPrefix(long, got) {
		t.Fatalf("shorten left %d bytes", len(got))
	}
	if got := shorten("bad: hello world", "hello world"); got != "bad: [prompt]" {
		t.Fatalf("shorten = %q", got)
	}
}
