# A Standalone Example for TensorRT5 C++ API

## Compile
g++ -std=c++11 sampleUffMNIST.cpp common.cpp -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lnvinfer -lnvparsers


