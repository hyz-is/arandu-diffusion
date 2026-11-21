//go:build unix

package diffusion

import (
	"runtime"
	"syscall"
)

// peakRSS is the high-water mark of the process's resident memory in bytes.
// getrusage reports kilobytes on Linux and bytes on macOS.
func peakRSS() uint64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil || ru.Maxrss < 0 {
		return 0
	}
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		return uint64(ru.Maxrss)
	}
	return uint64(ru.Maxrss) * 1024
}
