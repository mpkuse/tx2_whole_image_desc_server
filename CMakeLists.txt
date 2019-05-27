
cmake_minimum_required (VERSION 2.8)
project (tx2_whole_image_desc_server_standalone)
add_definitions(-std=c++11) 

add_executable( helloDemo src_tensorrt_example/hello.cpp )
target_link_libraries( helloDemo )



include_directories( "/usr/local/cuda-9.0/include" )
add_executable( tensorrt_demo 
	src_tensorrt_example/inference.cpp 
	src_tensorrt_example/TFRTEngine.cpp
	src_tensorrt_example/common.cpp
)
target_link_libraries( tensorrt_demo 
	/usr/lib/aarch64-linux-gnu/libnvinfer.so
	 /usr/lib/aarch64-linux-gnu/libcudnn.so
	/usr/local/cuda-9.0/targets/aarch64-linux/lib/libcudart.so
	/usr/lib/aarch64-linux-gnu/libnvparsers.so
)
