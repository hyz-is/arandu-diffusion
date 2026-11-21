//go:build sdcpp && sd_vulkan

package diffusion

// Vulkan, built with `make archives BACKEND=vulkan`.

/*
#cgo CFLAGS: -I${SRCDIR}/cgo_headers
#cgo LDFLAGS: -L${SRCDIR}/lib -Wl,--start-group -lstable-diffusion -lggml -lggml-cpu -lggml-base -lggml-vulkan -Wl,--end-group -lvulkan -lstdc++ -lm -lpthread
*/
import "C"

const linkedBackend = "vulkan"
