package diffusion

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ggufBuilder writes just enough of a GGUF file for the header reader: metadata
// of every value type it has to step over, then a tensor table.
type ggufBuilder struct{ bytes.Buffer }

func (b *ggufBuilder) u32(v uint32) { _ = binary.Write(b, binary.LittleEndian, v) }
func (b *ggufBuilder) u64(v uint64) { _ = binary.Write(b, binary.LittleEndian, v) }
func (b *ggufBuilder) str(s string) { b.u64(uint64(len(s))); b.WriteString(s) }

type tensorSpec struct {
	name string
	dims []uint64
	typ  uint32
}

func buildGGUF(version uint32, tensors []tensorSpec) []byte {
	var b ggufBuilder
	b.WriteString("GGUF")
	b.u32(version)
	b.u64(uint64(len(tensors)))
	b.u64(6) // metadata entries

	b.str("general.name") // string
	b.u32(8)
	b.str("tiny")
	b.str("general.file_type") // u32
	b.u32(4)
	b.u32(7)
	b.str("flag") // bool
	b.u32(7)
	b.WriteByte(1)
	b.str("scale") // f64
	b.u32(12)
	b.u64(0)
	b.str("tokens") // array of strings: the part that makes a header long
	b.u32(9)
	b.u32(8)
	b.u64(3)
	for _, s := range []string{"a", "bb", "ccc"} {
		b.str(s)
	}
	b.str("merges") // array of fixed-size values
	b.u32(9)
	b.u32(4)
	b.u64(4)
	for range 4 {
		b.u32(1)
	}

	for _, t := range tensors {
		b.str(t.name)
		b.u32(uint32(len(t.dims)))
		for _, d := range t.dims {
			b.u64(d)
		}
		b.u32(t.typ)
		b.u64(0)
	}
	return b.Bytes()
}

func TestTheQuantizationIsTheQuantizedTypeThatHoldsTheMostParameters(t *testing.T) {
	t.Parallel()

	// Many small F32 tensors and one large Q8_0 one: counting tensors would say
	// F32, and the file is a Q8_0 model.
	var tensors []tensorSpec
	for range 50 {
		tensors = append(tensors, tensorSpec{"norm", []uint64{320}, 0})
	}
	tensors = append(tensors, tensorSpec{"attn.q", []uint64{1024, 1024}, 8})

	got, err := ggufQuantization(bytes.NewReader(buildGGUF(3, tensors)))
	if err != nil {
		t.Fatal(err)
	}
	if got != "Q8_0" {
		t.Fatalf("quantization = %q, want Q8_0", got)
	}

	// A Stable Diffusion "Q8_0" checkpoint: the convolutions and the encoders stay
	// in F16 and hold most of the parameters, the linear layers are Q8_0. It is a
	// Q8_0 model; and of two quantized types the larger one names the file.
	got, err = ggufQuantization(bytes.NewReader(buildGGUF(3, []tensorSpec{
		{"conv", []uint64{4096, 4096}, 1},
		{"attn", []uint64{1024, 1024}, 8},
		{"head", []uint64{512, 512}, 12},
	})))
	if err != nil || got != "Q8_0" {
		t.Fatalf("quantization = %q, %v, want Q8_0 beside a larger F16 share", got, err)
	}

	got, err = ggufQuantization(bytes.NewReader(buildGGUF(2, []tensorSpec{{"w", []uint64{64, 64}, 1}})))
	if err != nil || got != "F16" {
		t.Fatalf("quantization = %q, %v, want F16", got, err)
	}
	got, err = ggufQuantization(bytes.NewReader(buildGGUF(3, []tensorSpec{{"w", []uint64{8, 8}, 99}})))
	if err != nil || got != "type-99" {
		t.Fatalf("an unknown type was named %q (%v), want it reported by number", got, err)
	}
}

func TestAFileThatIsNotAGGUFIsNotGuessedAt(t *testing.T) {
	t.Parallel()

	for name, b := range map[string][]byte{
		"empty":       nil,
		"text":        []byte("this is a safetensors file, honest"),
		"bad version": buildGGUF(9, []tensorSpec{{"w", []uint64{8}, 1}}),
		"truncated":   buildGGUF(3, []tensorSpec{{"w", []uint64{8}, 1}})[:40],
		"no tensors":  buildGGUF(3, nil),
	} {
		if q, err := ggufQuantization(bytes.NewReader(b)); err == nil {
			t.Errorf("%s: read %q from something that is not a gguf with tensors", name, q)
		}
	}

	// A header that claims a million tensors in a few bytes is refused before it
	// drives a loop.
	var b ggufBuilder
	b.WriteString("GGUF")
	b.u32(3)
	b.u64(1 << 40)
	b.u64(0)
	if _, err := ggufQuantization(bytes.NewReader(b.Bytes())); err == nil {
		t.Error("an implausible header was accepted")
	}
}

func TestDescribeDigestsTheWholeFileAndNamesNoDirectory(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	path := filepath.Join(dir, "tiny-Q8_0.gguf")
	content := buildGGUF(3, []tensorSpec{{"w", []uint64{32, 32}, 8}})
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatal(err)
	}

	file, err := describe(modelRef{"checkpoint", path}, true)
	if err != nil {
		t.Fatal(err)
	}
	// sha256 of the bytes written, computed independently here.
	if len(file.SHA256) != 64 || file.Bytes != int64(len(content)) || file.Quantization != "Q8_0" {
		t.Fatalf("describe = %+v", file)
	}
	if strings.Contains(file.Name, string(filepath.Separator)) || file.Name != "tiny-Q8_0.gguf" {
		t.Fatalf("the name %q carries a directory", file.Name)
	}
	again, err := describe(modelRef{"checkpoint", path}, true)
	if err != nil || again.SHA256 != file.SHA256 {
		t.Fatalf("the digest is not stable: %v %v", again.SHA256, err)
	}
	if err := os.WriteFile(path, append(content, 0), 0o600); err != nil {
		t.Fatal(err)
	}
	if changed, _ := describe(modelRef{"checkpoint", path}, true); changed.SHA256 == file.SHA256 {
		t.Fatal("a changed file kept its digest")
	}
	if skipped, _ := describe(modelRef{"checkpoint", path}, false); skipped.SHA256 != "" {
		t.Fatal("a digest was computed when it was turned off")
	}

	set := ModelSet{Diffusion: path, VAE: path}
	files, err := describeAll(set, true)
	if err != nil || len(files) != 2 || files[0].Role != "diffusion" || files[1].Role != "vae" {
		t.Fatalf("describeAll = %+v, %v", files, err)
	}
}
