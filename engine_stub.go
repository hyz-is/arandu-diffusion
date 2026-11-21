//go:build !sdcpp

package diffusion

// Load is the real engine, and this build does not have it. Build with
// -tags sdcpp after producing the archives (see the README); without the tag
// the package compiles anywhere, with no C toolchain, and an application that
// only needs Fake or the recorded runs pays nothing for the library.
func Load(set ModelSet, opts ...Option) (Engine, error) {
	if err := set.Validate(); err != nil {
		return nil, err
	}
	if _, err := newLoadOptions(opts); err != nil {
		return nil, err
	}
	return nil, ErrBackendUnavailable
}

// Linked reports whether this build links stable-diffusion.cpp.
const Linked = false
