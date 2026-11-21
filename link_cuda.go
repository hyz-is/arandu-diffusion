//go:build sdcpp && sd_cuda

package diffusion

// NVIDIA GPU, built with `make archives BACKEND=cuda CUDA_ARCHITECTURES=<cc>`.
// The architecture must match the card: a build for a higher one loads, starts
// and dies at the first kernel launch.
//
// /usr/local/cuda/lib64/stubs lets -lcuda resolve on a build host with no GPU;
// the real libcuda arrives at run time with the driver.

/*
#cgo CFLAGS: -I${SRCDIR}/cgo_headers
#cgo LDFLAGS: -L${SRCDIR}/lib -Wl,--start-group -lstable-diffusion -lggml -lggml-cpu -lggml-base -lggml-cuda -Wl,--end-group -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread
*/
import "C"

const linkedBackend = "cuda"
