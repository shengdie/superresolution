cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
project(SRmodel)

set(SOURCE_FILES
                    #state_loader.hpp
                    #state_loader.cpp
                    model_cppall.hpp
                    model_cppall.cpp
                    videosr.cpp)

# add_executable(model model_cppall.cpp)
add_executable(model ${SOURCE_FILES})
# set_property(TARGET model PROPERTY CXX_STANDARD 20)
list(APPEND CMAKE_PREFIX_PATH "./libtorch")

find_package(Torch REQUIRED)

# add_executable(example-app example-app.cpp)
target_link_libraries(model "${TORCH_LIBRARIES}")
set_property(TARGET model PROPERTY CXX_STANDARD 14)

#include_directories(${CMAKE_SOURCE_DIR}/rapidjson/include)
if(UNIX AND NOT APPLE)
    #set(LINUX TRUE)
    #include_directories(${CMAKE_SOURCE_DIR}/../third_party/json/include)
    message(STATUS ">>> Linux:")
    # link cuda if it is available
    find_package(CUDA)
    #if(CUDA_FOUND)
    target_link_libraries(model ${CUDA_LIBRARIES})
   
    list(APPEND CMAKE_PREFIX_PATH "/workspace/locallibs")
    find_package( OpenCV REQUIRED )
    include_directories( ${OpenCV_INCLUDE_DIRS} )

    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

    link_directories(${OpenCV_LIBRARY_DIRS})
    set(OpenCV_LIBS opencv_core opencv_video opencv_videoio opencv_imgcodecs)
    target_link_libraries( model ${OpenCV_LIBS} )

endif() 


#target_link_libraries( model "opencv_core;opencv_highgui;")


# target_link_libraries(webStream Qt5::Widgets)
# target_link_libraries(webStream Qt5::Core)
#endif()
