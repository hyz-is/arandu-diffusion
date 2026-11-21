//go:build sdcpp && !sd_metal && !sd_cuda && !sd_vulkan

package diffusion

// The CPU build: the four archives `make archives` leaves in lib/.
//
// The package declares -L${SRCDIR}/lib, which for a dependency resolves to the
// module cache -- read-only and shared by every build on the machine. An
// application points the linker at its own copy of the archives with a second
// -L in CGO_LDFLAGS; the linker searches every directory it is given.

/*
#cgo CFLAGS: -I${SRCDIR}/cgo_headers
#cgo darwin LDFLAGS: -L${SRCDIR}/lib -lstable-diffusion -lggml -lggml-cpu -lggml-base -lc++ -lm
#cgo !darwin LDFLAGS: -L${SRCDIR}/lib -Wl,--start-group -lstable-diffusion -lggml -lggml-cpu -lggml-base -Wl,--end-group -lstdc++ -lm -lpthread
*/
import "C"

const linkedBackend = "cpu"
