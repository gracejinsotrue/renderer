# makefile for 3D Engine

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall -Wextra
INCLUDES = -I/usr/include/SDL2
LIBS = -lSDL2 -lSDL2main -lm

# Source files
SOURCES = main.cpp Engine.cpp model.cpp our_gl.cpp tgaimage.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = engine

# default target
all: $(TARGET)

#linking the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LIBS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install SDL2 (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install libsdl2-dev

# Install SDL2 (macOS with Homebrew)
install-deps-mac:
	brew install sdl2

# Run the program
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Create obj directory if it doesn't exist
setup:
	mkdir -p obj

# Capture a rotating animation (360 frames = full rotation)
capture-rotation: $(TARGET)
	./$(TARGET) --capture-sequence rotation 360 10.0

.PHONY: all clean install-deps install-deps-mac run debug setup capture-rotation

# Dependencies (auto-generated with `g++ -MM *.cpp`)
Engine.o: Engine.cpp Engine.h geometry.h model.h tgaimage.h our_gl.h
main.o: main.cpp Engine.h geometry.h model.h tgaimage.h our_gl.h
model.o: model.cpp model.h geometry.h tgaimage.h
our_gl.o: our_gl.cpp our_gl.h tgaimage.h geometry.h
tgaimage.o: tgaimage.cpp tgaimage.h