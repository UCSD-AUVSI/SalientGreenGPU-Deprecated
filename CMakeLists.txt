project( SalientGreenGPU )

cmake_minimum_required(VERSION 2.8)

include_directories(include)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules)

######################################################################
# OpenCV 2.2
######################################################################
find_package(OpenCV2 QUIET REQUIRED)
if(OpenCV2_FOUND)
  set(LIBS ${LIBS} ${OpenCV2_LIBRARIES})
  link_libraries(${OpenCV2_LIBRARIES})
  message(STATUS "Found OpenCV2 headers at ${OpenCV2_INCLUDE_DIRS}")
  message(STATUS "Found OpenCV2 libraries at ${OpenCV2_LIBRARIES}")
else(OpenCV2_FOUND)
  message(STATUS "*** Missing OpenCV2 ***")
endif(OpenCV2_FOUND)
include_directories(/usr/local/include)

# Build the CUDA components as a shared library
find_package(CUDA 4.0 QUIET REQUIRED)

message(STATUS "Make sure you tell CUDA where to find an appropriate compiler with --compiler-bindir=path in the
COMPILER_BINDIR variable")

set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;")
if( COMPILER_BINDIR )
	set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --compiler-bindir=${COMPILER_BINDIR})
endif( COMPILER_BINDIR )

CUDA_INCLUDE_DIRECTORIES( include )
CUDA_ADD_LIBRARY( cudaCode src/SalientGreenGPU/Cuda/dcAdjust.cu SHARED )

link_libraries(cudaCode)

# Tell cmake to install this lib:
install(TARGETS cudaCode LIBRARY DESTINATION lib COMPONENT libs)


# Non CUDA Stuff follows
set(CXX_FLAGS -std=c++0x -Wall -Werror -Wno-deprecated -fPIC -g -O4)
add_definitions(${CXX_FLAGS})

SET(CMAKE_C_COMPILER /opt/local/bin/g++-mp-4.6)
SET(CMAKE_CXX_COMPILER /opt/local/bin/g++-mp-4.6)

#Be sure to list all of the necessary components here



CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/SalientGreenGPU/config.h)

file(GLOB_RECURSE SOURCE_FILES src/SalientGreenGPU/*.C src/SalientGreenGPU/details/*.C)

#add_executable(nrtDesigner src/Apps/nrtDesigner/main.C ${DESIGNER_MOC_HEADERS} ${DESIGNER_SRCS})
#target_link_libraries(nrtDesigner ${QT_LIBRARIES})

add_executable( SalientGreenGPU ${SOURCE_FILES} )
target_link_libraries( SalientGreenGPU ${LIBS} )
