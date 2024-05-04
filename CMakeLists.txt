cmake_minimum_required(VERSION 3.1)
project(torchTest)

# Set the path to the installation of LibTorch

set(LIBTORCH_PATH /home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch)

# Set the path to the NVToolsExt library

set(LIBNVTOOLSEXT "/home/frackowiak/Documents/INCLUDES/libtorch-cxx11-abi-shared-with-deps-2.3.0+cu121/libtorch/lib/libnvToolsExt-847d78f2.so.1")

# Find PyTorch

find_package(Torch REQUIRED PATHS ${LIBTORCH_PATH})

# Add PyTorch include directories to the project

include_directories(${TORCH_INCLUDE_DIRS})

# Create an executable with the main.cpp source file

add_executable(${PROJECT_NAME} model.cpp)

# Link against PyTorch libraries

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${LIBNVTOOLSEXT})