package diffusion

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// describe reads one model file: its size, its digest and, for a GGUF, its
// quantization. The digest is what ties a run to the exact weights; a name is
// not enough, because a file can be replaced under the same name.
func describe(ref modelRef, digest bool) (ModelFile, error) {
	f, err := os.Open(ref.path)
	if err != nil {
		return ModelFile{}, fmt.Errorf("%w: %s: %v", ErrInvalidModelSet, ref.role, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return ModelFile{}, fmt.Errorf("%w: %s: %v", ErrInvalidModelSet, ref.role, err)
	}
	file := ModelFile{Role: ref.role, Name: filepath.Base(ref.path), Bytes: info.Size()}
	if q, err := ggufQuantization(f); err == nil {
		file.Quantization = q
	}
	if digest {
		if _, err := f.Seek(0, io.SeekStart); err != nil {
			return ModelFile{}, err
		}
		h := sha256.New()
		if _, err := io.Copy(h, f); err != nil {
			return ModelFile{}, fmt.Errorf("%w: %s: hashing: %v", ErrInvalidModelSet, ref.role, err)
		}
		file.SHA256 = hex.EncodeToString(h.Sum(nil))
	}
	return file, nil
}

// describeAll describes every file of the set concurrently, since hashing is
// bound by the disk and the files are independent. The order of the answer is
// the order of the set.
func describeAll(set ModelSet, digest bool) ([]ModelFile, error) {
	refs := set.files()
	out := make([]ModelFile, len(refs))
	errs := make([]error, len(refs))
	var wg sync.WaitGroup
	for i, ref := range refs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			out[i], errs[i] = describe(ref, digest)
		}()
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}
	return out, nil
}

// ggmlTypeNames maps the tensor types of a GGUF file to their names. The numbers
// are ggml's and have been stable since the format was published; a type this
// table does not know is reported by number, never guessed.
var ggmlTypeNames = map[uint32]string{
	0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
	10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
	16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
	22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
	29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
}

// ggufQuantization returns the weight type a GGUF file is quantized to: the
// quantized type that holds the most parameters, or, when the file has none, the
// plain type that does (F16, BF16, F32).
//
// Counting parameters rather than tensors matters, because the norms and biases
// stay in F32 whatever the file is called. And preferring a quantized type over
// a larger float one matters too: a Stable Diffusion "Q8_0" checkpoint quantizes
// the linear layers of the diffusion model and leaves the convolutions, the
// text encoder and the VAE in F16, and F16 holds most of its parameters. It is a
// Q8_0 model, and that is what the name and the comparison across models need.
// It reads the header and the tensor table and nothing else.
func ggufQuantization(r io.Reader) (string, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	var magic [4]byte
	if _, err := io.ReadFull(br, magic[:]); err != nil || string(magic[:]) != "GGUF" {
		return "", errors.New("not a gguf file")
	}
	var version uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil || version < 2 || version > 3 {
		return "", errors.New("unsupported gguf version")
	}
	var tensors, kvs uint64
	if err := binary.Read(br, binary.LittleEndian, &tensors); err != nil {
		return "", err
	}
	if err := binary.Read(br, binary.LittleEndian, &kvs); err != nil {
		return "", err
	}
	// A header that claims more than a real file ever holds is refused before it
	// drives a loop.
	if tensors > 1<<20 || kvs > 1<<20 {
		return "", errors.New("implausible gguf header")
	}
	for range kvs {
		if _, err := readString(br); err != nil {
			return "", err
		}
		var vt uint32
		if err := binary.Read(br, binary.LittleEndian, &vt); err != nil {
			return "", err
		}
		if err := skipValue(br, vt, 0); err != nil {
			return "", err
		}
	}
	params := map[uint32]uint64{}
	for range tensors {
		if _, err := readString(br); err != nil {
			return "", err
		}
		var dims uint32
		if err := binary.Read(br, binary.LittleEndian, &dims); err != nil || dims > 8 {
			return "", errors.New("bad tensor rank")
		}
		n := uint64(1)
		for range dims {
			var d uint64
			if err := binary.Read(br, binary.LittleEndian, &d); err != nil {
				return "", err
			}
			n *= d
		}
		var typ uint32
		var offset uint64
		if err := binary.Read(br, binary.LittleEndian, &typ); err != nil {
			return "", err
		}
		if err := binary.Read(br, binary.LittleEndian, &offset); err != nil {
			return "", err
		}
		params[typ] += n
	}
	if len(params) == 0 {
		return "", errors.New("no tensors")
	}
	best := func(only func(uint32) bool) (uint32, bool) {
		var pick uint32
		found := false
		for t, n := range params {
			if !only(t) {
				continue
			}
			if !found || n > params[pick] || (n == params[pick] && t < pick) {
				pick, found = t, true
			}
		}
		return pick, found
	}
	pick, ok := best(isQuantized)
	if !ok {
		pick, _ = best(func(uint32) bool { return true })
	}
	if name, ok := ggmlTypeNames[pick]; ok {
		return name, nil
	}
	return fmt.Sprintf("type-%d", pick), nil
}

// isQuantized reports whether a ggml type stores weights in blocks of fewer bits:
// everything that is not a plain float or integer.
func isQuantized(t uint32) bool {
	switch t {
	case 0, 1, 24, 25, 26, 27, 28, 30: // F32 F16 I8 I16 I32 I64 F64 BF16
		return false
	}
	return true
}

func readString(r io.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	if n > 1<<24 {
		return "", errors.New("implausible gguf string")
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

// skipValue steps over one metadata value of the given type. Arrays nest, so the
// depth is bounded.
func skipValue(r *bufio.Reader, vt uint32, depth int) error {
	if depth > 4 {
		return errors.New("gguf array nested too deep")
	}
	fixed := map[uint32]int64{0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
	if n, ok := fixed[vt]; ok {
		_, err := r.Discard(int(n))
		return err
	}
	switch vt {
	case 8:
		var n uint64
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return err
		}
		return discard(r, n)
	case 9:
		var et uint32
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &et); err != nil {
			return err
		}
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return err
		}
		if n, ok := fixed[et]; ok {
			return discard(r, count*uint64(n))
		}
		for range count {
			if err := skipValue(r, et, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	return fmt.Errorf("unknown gguf value type %d", vt)
}

// discard skips n bytes, which may exceed what Discard takes at once.
func discard(r *bufio.Reader, n uint64) error {
	for n > 0 {
		step := n
		if step > 1<<30 {
			step = 1 << 30
		}
		d, err := r.Discard(int(step))
		n -= uint64(d)
		if err != nil {
			return err
		}
	}
	return nil
}
