//go:build sdcpp && sd_metal

package diffusion

// Apple GPU, built with `make archives BACKEND=metal`.

/*
#cgo CFLAGS: -I${SRCDIR}/cgo_headers
#cgo LDFLAGS: -L${SRCDIR}/lib -lstable-diffusion -lggml -lggml-cpu -lggml-base -lggml-metal -lc++ -lm -framework Metal -framework MetalKit -framework Foundation -framework Accelerate
*/
import "C"

const linkedBackend = "metal"
