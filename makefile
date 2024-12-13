# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -I/opt/homebrew/Cellar/opencv/4.10.0_12/include/opencv4

# Linker flags
LDFLAGS = -L/opt/homebrew/Cellar/opencv/4.10.0_12/lib -I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_viz -lopencv_xfeatures2d

# Target executable
TARGET = bin/app

SRC = src/main_initial_dense.cpp src/bfmatcher.cpp src/epipolar.cpp src/orb.cpp src/ransac.cpp src/triangulation.cpp src/bundle.cpp

# Build target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Clean build files
clean:
	rm -f $(TARGET)
