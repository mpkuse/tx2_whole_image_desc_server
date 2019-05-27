#include <iostream> 
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "NvUtils.h"

using namespace nvuffparser; 
using namespace nvinfer1;

int main()
{
std::cout << "Hello World\n";
auto parser = createUffParser(); 
}
