# Builds the static archives of stable-diffusion.cpp that the `sdcpp` build tag
# links. The module ships no compiled code: everything below is produced from
# the pinned submodule into lib/, which is git-ignored.
#
#   git submodule update --init stable-diffusion.cpp
#   git -C stable-diffusion.cpp submodule update --init ggml
#   make archives                    # CPU only, the default
#   make archives BACKEND=metal      # Apple silicon
#   make archives BACKEND=cuda CUDA_ARCHITECTURES=75
#   make archives BACKEND=vulkan
#
# Then, from the application:
#
#   CGO_ENABLED=1 CGO_LDFLAGS="-L/path/to/lib" go build -tags sdcpp ./...
#
# A GPU backend adds its own tag on top of sdcpp: sd_metal, sd_cuda, sd_vulkan.

.PHONY: archives clean headers check-headers

SD_DIR      := $(abspath stable-diffusion.cpp)
BUILD_DIR   := $(abspath build)
LIB_DIR     := $(abspath lib)
BACKEND     ?= cpu
JOBS        ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)
CUDA_ARCHITECTURES ?= native

# What every build shares. Image I/O codecs beyond PNG/JPEG/BMP (WebP, WebM) are
# off: the Go side hands the library raw tensors and encodes the PNG itself, so
# the library never reads or writes a file format. Accelerate and OpenMP are off
# for the same reason of staying self-contained: no framework and no libgomp to
# name in the link line of a CPU build.
COMMON := -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	-DSD_BUILD_EXAMPLES=OFF -DSD_BUILD_SHARED_LIBS=OFF \
	-DSD_WEBP=OFF -DSD_WEBM=OFF \
	-DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF

ifeq ($(BACKEND),cpu)
FLAGS := $(COMMON) -DSD_METAL=OFF -DGGML_METAL=OFF -DSD_CUDA=OFF -DSD_VULKAN=OFF
else ifeq ($(BACKEND),metal)
FLAGS := $(COMMON) -DSD_METAL=ON
else ifeq ($(BACKEND),cuda)
FLAGS := $(COMMON) -DSD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHITECTURES)
else ifeq ($(BACKEND),vulkan)
FLAGS := $(COMMON) -DSD_VULKAN=ON
else
$(error BACKEND is "$(BACKEND)": cpu, metal, cuda or vulkan)
endif

archives:
	@test -f $(SD_DIR)/CMakeLists.txt || { echo "run: git submodule update --init stable-diffusion.cpp"; exit 1; }
	@test -f $(SD_DIR)/ggml/CMakeLists.txt || { echo "run: git -C stable-diffusion.cpp submodule update --init ggml"; exit 1; }
	cmake -S $(SD_DIR) -B $(BUILD_DIR)/$(BACKEND) $(FLAGS)
	cmake --build $(BUILD_DIR)/$(BACKEND) --target stable-diffusion -j $(JOBS)
	rm -rf $(LIB_DIR) && mkdir -p $(LIB_DIR)
	find $(BUILD_DIR)/$(BACKEND) -name 'lib*.a' -not -path '*/CMakeFiles/*' -exec cp {} $(LIB_DIR)/ \;
	@echo "archives in $(LIB_DIR):" && ls $(LIB_DIR)

# cgo_headers/ mirrors the one public header, because the Go module proxy strips
# git submodules from the module zip. check-headers is what CI runs.
headers:
	cp $(SD_DIR)/include/stable-diffusion.h cgo_headers/stable-diffusion.h

check-headers:
	cmp $(SD_DIR)/include/stable-diffusion.h cgo_headers/stable-diffusion.h

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
