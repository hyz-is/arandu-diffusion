//go:build !unix

package diffusion

// peakRSS has no source on this platform and says so with zero.
func peakRSS() uint64 { return 0 }
