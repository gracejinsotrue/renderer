# makefile for building enigne

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -I/usr/include/SDL2
LIBS = -lSDL2 -lSDL2main -lm

# source files (includes ray tracing and scene graph)
SOURCES = main.cpp Engine.cpp model.cpp our_gl.cpp tgaimage.cpp geometry.cpp Scene.cpp SceneNode.cpp ray_tracer_integration.cpp raytracing.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = unified_engine

# default target
all: $(TARGET)

#link executable
$(TARGET): $(OBJECTS) cuda_triangle.o
	$(CXX) $(OBJECTS) cuda_triangle.o -o $(TARGET) $(LIBS) -L/usr/local/cuda/lib64 -lcudart

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

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

.PHONY: all clean install-deps install-deps-mac run debug setup capture-rotation

# dependencies
Engine.o: Engine.cpp Engine.h geometry.h model.h tgaimage.h our_gl.h Scene.h SceneNode.h ray_tracer_integration.h bvh.h
main.o: main.cpp Engine.h geometry.h model.h tgaimage.h our_gl.h Scene.h SceneNode.h ray_tracer_integration.h
model.o: model.cpp model.h geometry.h tgaimage.h
our_gl.o: our_gl.cpp our_gl.h tgaimage.h geometry.h
tgaimage.o: tgaimage.cpp tgaimage.h
geometry.o: geometry.cpp geometry.h
Scene.o: Scene.cpp Scene.h SceneNode.h model.h
SceneNode.o: SceneNode.cpp SceneNode.h model.h
ray_tracer_integration.o: ray_tracer_integration.cpp ray_tracer_integration.h bvh.h unified_math.h