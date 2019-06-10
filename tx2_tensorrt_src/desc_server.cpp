
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>

#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"

using namespace nvuffparser;
using namespace nvinfer1;
#include "common.h"

using namespace nvuffparser;
using namespace nvinfer1;
using namespace std;
static Logger gLogger;


//-------------------------- Utils ---------------------------//
#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)



#define __loadModel_cout(msg) msg;
ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();

	__loadModel_cout( cout << "[loadModelAndCreateEngine] Parse UFF File: " << uffFile << endl; )
#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
        RETURN_AND_LOG(nullptr, ERROR, "Fail to parse");
    builder->setHalf2Mode(true);
#endif

    /* we create the engine */
    __loadModel_cout( cout << "[loadModelAndCreateEngine] We create(compile) the engine with batchsize=" << maxBatchSize << " and workspacesize=" << MAX_WORKSPACE << endl;)
    __loadModel_cout( cout << "[loadModelAndCreateEngine] Depending on model size this usually takes 10sec. TODO: Compile and store the engine file\n");
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
        RETURN_AND_LOG(nullptr, ERROR, "Unable to create engine. Perhaps run this executable as root");

    /* we can clean the network and the parser */
    __loadModel_cout( cout << "[loadModelAndCreateEngine] Engine creating successfully. Cleanup the parser\n"; )
    network->destroy();
    builder->destroy();

    return engine;
}


ICudaEngine * loadCudaEngine( const char * engine_fname )
{
  IRuntime* runtime = createInferRuntime(gLogger);
  __loadModel_cout(cout << "[loadCudaEngine] Load Engine File: " << engine_fname << endl;)
  std::ifstream gieModelStream(engine_fname, ios::binary);
  if( !gieModelStream ) { cout << "[loadCudaEngine] ERROR cannot open file: " << engine_fname << endl; return nullptr; }

  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);

  // allocate ram memory to hold the .engine file
  __loadModel_cout(cout << "[loadCudaEngine] allocate " << modelSize << " bytes " << endl;)
  void* modelMem = malloc(modelSize);
  if( !modelMem ) {
      printf("[loadCudaEngine] failed to allocate %i bytes to deserialize model\n", modelSize);
      return nullptr;
  }

  gieModelStream.read((char*)modelMem, modelSize);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
  __loadModel_cout( cout << "[loadCudaEngine] deserializeCudaEngine successfully. Cleanup the parser\n"; )
  return engine;

}
//-------------------------- END Utils ---------------------------//

//-----------------------------execution demo -------------------------//
void demo_exec( ICudaEngine& engine )
{
  const int IM_ROWS = 240;
  const int IM_COLS = 320;
  const int IM_CHNLS = 3;
  const int OUTPUT_SZ = 1*30*40*256;
	float * input = new float[IM_ROWS*IM_COLS*IM_CHNLS];

	// uint8_t * rawpgm = new uint8_t[28*28];
    // readPGMFile( "data/0.pgm", rawpgm, 28, 28);
	for( int i=0; i<IM_ROWS*IM_COLS*IM_CHNLS; i++ ) {
		// input[i]=1.0 - float(rawpgm[i])/255.;
    input[i] = 0.2;
  }

	float * output = new float[OUTPUT_SZ];


	cout << "[demo_exec]Start\n";
	// Execution context
    IExecutionContext* context = engine.createExecutionContext();

    // Bindings
    int nbinds = engine.getNbBindings();
    cout << "nbinds = " << nbinds << endl;

    int inputIndex = engine.getBindingIndex( "input_1" );
    int outputIndex = engine.getBindingIndex( "conv_pw_5_relu/Relu6" );
    cout << "engine.getBindingIndex( \"input_1\" ) ---> "<< inputIndex << endl; //0
    cout << "engine.getBindingIndex( \"conv_pw_5_relu/Relu6\"  ) ---> "<< outputIndex << endl; //1


    assert( inputIndex>=0 && outputIndex>=0);

	// Create GPU buffers on device
    void * buffers[2];
	CHECK( cudaMalloc( &buffers[inputIndex], 1*IM_CHNLS*IM_ROWS*IM_COLS*sizeof(float)) );
	CHECK( cudaMalloc( &buffers[outputIndex], OUTPUT_SZ*sizeof(float)) );


	// Host --> Device
	cout << "cudaMemcpyHostToDevice\n" ;
	CHECK( cudaMemcpy( buffers[inputIndex], input, IM_ROWS*IM_COLS*IM_CHNLS*sizeof(float), cudaMemcpyHostToDevice ) );

	// Execute
	cout << "execute\n";
	            auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(1, &buffers[0]);
                auto t_end = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    cout << "Execution done in " << ms << " ms\n";

    // Device --> Host
	cout << "cudaMemcpyDeviceToHost\n";
	CHECK( cudaMemcpy( output, buffers[outputIndex], OUTPUT_SZ*sizeof(float), cudaMemcpyDeviceToHost ) );

	// Note:
	// 	A better way is to use DMA with Async memory copy and cuda streams. See sampleMNISTAPI.cpp to know how to do it.

	cout << "Output\n";
	for( int i=0 ; i<10; i++ ) {
		cout << i << ": " << output[i] << endl;
	}

	// Release
	cout << "Release\n";
	CHECK( cudaFree(buffers[inputIndex]));
	CHECK( cudaFree(buffers[outputIndex]));
	delete [] input;
	delete [] output;
}

//-----------------------------END execution demo -------------------------//


//-----------------------ROS ----------------------------------------------//
#include <ros/ros.h>
#include <tx2_whole_image_desc_server/WholeImageDescriptorCompute.h>
#include "TermColor.h"

bool handle(tx2_whole_image_desc_server::WholeImageDescriptorCompute::Request  &req,
         tx2_whole_image_desc_server::WholeImageDescriptorCompute::Response &res)
{
    ROS_INFO( "[::handle] Request Received");
    return true;
}

/// This class contains all the UFF related stuff. Roughly speaking it follows the main
/// of file standalone/bare_mnist.cpp
class UFFModelDescriptorServer
{
public:
  UFFModelDescriptorServer()
  {
      std::cout << "[UFFModelDescriptorServer] Constructor\n";
      if( init() == EXIT_FAILURE ) {
        exit(EXIT_FAILURE);
      }
  }

  ~UFFModelDescriptorServer()
  {
    std::cout << "[UFFModelDescriptorServer] Destructor\n";

    engine->destroy();
    shutdownProtobufLibrary();
  }

  bool handle(tx2_whole_image_desc_server::WholeImageDescriptorCompute::Request  &req,
           tx2_whole_image_desc_server::WholeImageDescriptorCompute::Response &res)
  {
      ROS_INFO( "[UFFModelDescriptorServer::handle] Request Received");
      demo_exec( *engine );
      return true;
  }

private:
  // Class globals
  ICudaEngine* engine = nullptr;


  int init()
  {
      int maxBatchSize = 1;
      auto parser = createUffParser();


      parser->registerInput("input_1", DimsCHW(3, 240, 320),  UffInputOrder::kNCHW);
      // parser->registerOutput("net_vlad_layer_1/Reshape_1");
      parser->registerOutput("conv_pw_5_relu/Relu6");
      // parser->registerOutput("block5_pool/MaxPool");


      // auto fileName = string("uff_ready/output_nvinfer.uff");
      auto fileName = string("/home/dji/catkin_ws/src/tx2_whole_image_desc_server/standalone/uff_ready/output_nvinfer.uff");
      engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);
      if (!engine) {
          RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
      }
      parser->destroy();


      // Allocate CUDA buffer for IO


  }
};

//-------------------- ROS Handle ---------------------------------------//

int main(int argc, char ** argv)
{
    //
    // Init RosNode
    std::cout << "============================================================\n";
    std::cout << "====WholeImageDescriptor Server for TX2 (Uses TensorRT)====\n";
    std::cout << "============================================================\n";
    ros::init(argc, argv, "tx2_whole_image_desc_server");
    ros::NodeHandle n;

    //
    // Setup UFFParser
    UFFModelDescriptorServer * obj = new UFFModelDescriptorServer();


    ros::ServiceServer service = n.advertiseService("whole_image_descriptor_compute", &UFFModelDescriptorServer::handle, obj);
    // ros::ServiceServer service = n.advertiseService("whole_image_descriptor_compute", handle );
    std::cout << TermColor::GREEN() << "whole_image_descriptor_compute is running" << TermColor::RESET() << endl;
    ros::spin();

    delete obj;
    std::cout << TermColor::RED() << "Exit TX2 image descriptor server\n" << TermColor::RESET() << endl;
    return 0;
}

//-----------------------END ROS -----------------------------------------//



#if 0
int main()
{
    auto fileName = string("uff_ready/output_nvinfer.uff"); //locateFile("data/lenet5.uff");
    std::cout << fileName << std::endl;

    int maxBatchSize = 1;
    auto parser = createUffParser();

    parser->registerInput("input_1", DimsCHW(3, 240, 320),  UffInputOrder::kNCHW);
    // parser->registerOutput("net_vlad_layer_1/Reshape_1");
    parser->registerOutput("conv_pw_5_relu/Relu6");
    // parser->registerOutput("block5_pool/MaxPool");

    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);
    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
    parser->destroy();

    //TODO : Execute
    demo_exec( *engine );
    //demo_exec_async( *engine );

    cout << "Execution finished\n";

    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
#endif
