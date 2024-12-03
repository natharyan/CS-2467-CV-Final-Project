# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -I/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4

# Linker flags
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.10.0_12/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_calib3d

# Target executable
TARGET = bin/app

SRC = src/main.cpp src/bfmatcher.cpp

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Clean build files
clean:
	rm -f $(TARGET)
