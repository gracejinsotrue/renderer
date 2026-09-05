# makefile for building enigne

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -MMD -MP
NVCC = /usr/local/cuda/bin/nvcc
# CUDA 12.0's nvcc refuses host gcc > 12, and the default g++ here is 14,
# so pin the host compiler. bump this if you upgrade the CUDA toolkit.
NVCC_CCBIN = g++-11
NVCCFLAGS = -std=c++17 -O2 -ccbin $(NVCC_CCBIN)
INCLUDES = -I/usr/include/SDL2
LIBS = -lSDL2 -lSDL2main -lm

# source files
SOURCES = main.cpp Engine.cpp model.cpp transform.cpp tgaimage.cpp geometry.cpp Scene.cpp SceneNode.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = unified_engine

# generated header dependencies, one .d per .o
DEPS = $(OBJECTS:.o=.d)

# default target
all: $(TARGET)

-include $(DEPS)

#link executable
$(TARGET): $(OBJECTS) cuda_triangle.o
	$(CXX) $(OBJECTS) cuda_triangle.o -o $(TARGET) $(LIBS) \
	  -L/usr/local/cuda/lib64 -lcudart

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


cuda_triangle.o: cuda_triangle.cu geometry.h tgaimage.h
	$(NVCC) $(NVCCFLAGS) -c cuda_triangle.cu -o cuda_triangle.o

# Standalone GPU tests, see tests/README.md. Run from src/.
# Regression tests: each asserts and exits non-zero on failure.
TESTS = test_cull test_blit test_mesh test_shaded test_frustum

# Local benchmarking and inspection tools. Not part of the regression
# suite and not tracked in git, so `tools` skips any that are absent.
TOOLS = test_wind bench bench_cull bench_tile profile_frame render_png
TESTOBJS = model.o tgaimage.o geometry.o transform.o cuda_triangle.o \
           Scene.o SceneNode.o
# The CPU reference rasterizer the differential tests compare against. It is
# test-only: compiled into the test binaries, never into the engine.
REFSRC = tests/reference_raster.cpp

tests: $(TESTOBJS)
	mkdir -p tests/bin
	for t in $(TESTS); do \
	  $(CXX) $(CXXFLAGS) -I. -Itests -o tests/bin/$$t tests/$$t.cpp $(REFSRC) $(TESTOBJS) \
	    -L/usr/local/cuda/lib64 -lcudart || exit 1; \
	done
	@echo "built: $(TESTS) -> tests/bin/"

tools: $(TESTOBJS)
	mkdir -p tests/bin
	for t in $(TOOLS); do \
	  [ -f tests/$$t.cpp ] || continue; \
	  $(CXX) $(CXXFLAGS) -I. -Itests -o tests/bin/$$t tests/$$t.cpp $(REFSRC) $(TESTOBJS) \
	    -L/usr/local/cuda/lib64 -lcudart || exit 1; \
	done

# Headless end-to-end test of the real Engine (needs SDL2 + the scene graph).
ENGOBJS = Engine.o Scene.o SceneNode.o model.o transform.o tgaimage.o \
          geometry.o cuda_triangle.o

test_ssao: $(ENGOBJS)
	mkdir -p tests/bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I. -o tests/bin/test_ssao \
	  tests/test_ssao.cpp $(ENGOBJS) $(LIBS) \
	  -L/usr/local/cuda/lib64 -lcudart

test_ssaa: $(ENGOBJS)
	mkdir -p tests/bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I. -o tests/bin/test_ssaa \
	  tests/test_ssaa.cpp $(ENGOBJS) $(LIBS) \
	  -L/usr/local/cuda/lib64 -lcudart

test_engine: $(ENGOBJS)
	mkdir -p tests/bin
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I. -o tests/bin/test_engine \
	  tests/test_engine.cpp $(ENGOBJS) $(LIBS) \
	  -L/usr/local/cuda/lib64 -lcudart

# Clean build files
clean:
	rm -f $(OBJECTS) $(DEPS) cuda_triangle.o $(TARGET)
	rm -rf tests/bin

# install SDL2 (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install libsdl2-dev

# install SDL2 (macOS with Homebrew)
install-deps-mac:
	brew install sdl2

#  run the program!
run: $(TARGET)
	./$(TARGET)

# debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)
setup:
	mkdir -p obj

# 
capture-rotation: $(TARGET)
	./$(TARGET) --capture-sequence rotation 360 10.0

.PHONY: all tests test_engine test_ssao test_ssaa clean install-deps install-deps-mac run debug setup capture-rotation

# header dependencies come from -MMD -MP via $(DEPS); there is no hand-written
# list to keep in sync.