cmake_minimum_required (VERSION 2.8)
project (proto_app)


include_directories( "/usr/local/cuda/include" )
add_executable( image_desc_try
    common.cpp
    image_desc_try.cpp
)

target_link_libraries( image_desc_try
   ${catkin_LIBRARIES}
   /usr/lib/aarch64-linux-gnu/libnvinfer.so
   /usr/lib/aarch64-linux-gnu/libnvparsers.so
   /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so
)



add_executable( imagenet
    common.cpp
    imagenet.cpp
)

target_link_libraries( imagenet
   ${catkin_LIBRARIES}
   /usr/lib/aarch64-linux-gnu/libnvinfer.so
   /usr/lib/aarch64-linux-gnu/libnvparsers.so
   /usr/local/cuda-10.0/targets/aarch64-linux/lib/libcudart.so
)
