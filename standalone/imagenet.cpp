
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>  //std::iota

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

#include "imagenet_labels.h"
//-------------------------- Utils ---------------------------//
#define MAX_WORKSPACE (1 << 30)

#define RETURN_AND_LOG(ret, severity, message)                                              \
    do {                                                                                    \
        std::string error_message = "sample_uff_mnist: " + std::string(message);            \
        gLogger.log(ILogger::Severity::k ## severity, error_message.c_str());               \
        return (ret);                                                                       \
    } while(0)

template <typename T>
std::vector<std::size_t> compute_order(const std::vector<T>& v)
{
	std::vector<std::size_t> indices(v.size());
	std::iota(indices.begin(), indices.end(), 0u);
	std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        return v[lhs] < v[rhs];
    });
    std::vector<std::size_t> res(v.size());
    for (std::size_t i = 0; i != indices.size(); ++i) {
    	res[indices[i]] = i;
    }
    return res;
}

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
    __loadModel_cout( cout << "[loadModelAndCreateEngine] We create the engine with batchsize=" << maxBatchSize << " and workspacesize=" << MAX_WORKSPACE << endl;)
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
  cout << "[loadCudaEngine] Load Engine File: " << engine_fname << endl;
  std::ifstream gieModelStream(engine_fname, ios::binary);
  if( !gieModelStream ) { cout << "[loadCudaEngine] ERROR cannot open file: " << engine_fname << endl; return nullptr; }

  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);

  // allocate ram memory to hold the .engine file
  cout << "[loadCudaEngine] allocate " << modelSize << " bytes " << endl;
  void* modelMem = malloc(modelSize);
  if( !modelMem ) {
      printf("[loadCudaEngine] failed to allocate %i bytes to deserialize model\n", modelSize);
      return nullptr;
  }

  gieModelStream.read((char*)modelMem, modelSize);
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
  return engine;

}
//-------------------------- END Utils ---------------------------//
void HWC_to_CHW( uint8_t * src, uint8_t * dst, int h, int w, int chnls )
{
  // src: in HWC format
  // dst : in CHW format

  for( int i=0 ; i<h ; i++ )
  {
    for( int j=0 ; j<w ; j++ )
    {
      for( int c=0 ; c<chnls ; c++ )
      {
        // src[ i*(w*chnls) + j*chnls + c ]
        // cout << i*(w*chnls) + j*chnls + c << "--->" << (h*w*c) + ( i*w + j ) << endl;
        dst[ (h*w*c) + ( i*w + j ) ] = src[ i*(w*chnls) + j*chnls + c ];
      }
    }
  }
}

int main12()
{
  const int IM_ROWS = 224;
  const int IM_COLS = 224;
  const int IM_CHNLS = 3;
  uint8_t * rawpgm_HWC = new uint8_t[IM_ROWS*IM_COLS*IM_CHNLS];
  uint8_t * rawpgm_CHW = new uint8_t[IM_CHNLS*IM_ROWS*IM_COLS];

  HWC_to_CHW(rawpgm_HWC, rawpgm_CHW, IM_ROWS, IM_COLS, IM_CHNLS );

}

void demo_exec( ICudaEngine& engine )
{
  const int IM_ROWS = 224;
  const int IM_COLS = 224;
  const int IM_CHNLS = 3;
  const int OUTPUT_SZ = 1000;
	float * input = new float[IM_ROWS*IM_COLS*IM_CHNLS];

	uint8_t * rawpgm = new uint8_t[IM_ROWS*IM_COLS*IM_CHNLS];
  uint8_t * rawpgm_chw = new uint8_t[IM_ROWS*IM_COLS*IM_CHNLS];
  readPGMFileP6( "../data/dog.pgm", rawpgm, IM_ROWS, IM_COLS);
  HWC_to_CHW( rawpgm, rawpgm_chw, IM_ROWS, IM_COLS, IM_CHNLS );
	for( int i=0; i<IM_ROWS*IM_COLS*IM_CHNLS; i++ ) {
		// input[i]=1.0 - float(rawpgm[i])/255.;
    // input[i] = 120.;
    // input[i] = (float( rawpgm[i] ) - 128.)*2.0/255.;
    input[i] = (float( rawpgm_chw[i] ) - 128.)*2.0/255.; //for mobilenet
    // input[i] = float( rawpgm_chw[i] )  ; //for vgg16
    // cout << input[i] << endl;

  }
  // cout << endl;
  delete [] rawpgm;
  delete [] rawpgm_chw;

	float * output = new float[OUTPUT_SZ];


	cout << "[demo_exec]Start\n";
	// Execution context
    IExecutionContext* context = engine.createExecutionContext();

    // Bindings
    int nbinds = engine.getNbBindings();
    cout << "nbinds = " << nbinds << endl;

    int inputIndex = engine.getBindingIndex( "input_1" );
    // int outputIndex = engine.getBindingIndex( "fc1000/BiasAdd" );
    // int outputIndex = engine.getBindingIndex( "reshape_2/Reshape" );
    // int outputIndex = engine.getBindingIndex( "predictions/Softmax" ); //vgg16
    int outputIndex = engine.getBindingIndex( "Logits/Softmax" ); //mobilenetv2

    cout << "engine.getBindingIndex( \"input_1\" ) ---> "<< inputIndex << endl; //0
    cout << "engine.getBindingIndex( \"Logits/Softmax\"  ) ---> "<< outputIndex << endl; //1
    // cout << "engine.getBindingIndex( \"predictions/Softmax\"  ) ---> "<< outputIndex << endl; //1


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


  // find top-5 predictions
  std::vector<float> my_vector {output, output + 1000};
  ImageNetLabels imagenet_labels;
  const auto order = compute_order<float>( my_vector );
  cout << "TopN:\n";
  for( int i=0 ; i<1000 ; i++ ) {
    if( order[i] >  995 )
      cout  << "order(i)=" << order[i] << "\t" << i << " : prob=" << output[i] << "\t" << imagenet_labels.imagenet_labelstring( i ) << endl;
  }


	// Release
	cout << "Release\n";
	CHECK( cudaFree(buffers[inputIndex]));
	CHECK( cudaFree(buffers[outputIndex]));
	delete [] input;
	delete [] output;
}


int main()
{
    //
    // Load UFF, parse it and serialize to .engine file
    //
    #if 1
    // auto fileName = string("../uff_ready/mobilenet_imagenet.uff");
    // auto fileName = string("../uff_ready/vgg16_imagenet.uff");
    auto fileName = string("../uff_ready/mobilenetv2_imagenet.uff");
    std::cout << fileName << std::endl;

    int maxBatchSize = 1;
    auto parser = createUffParser();

    parser->registerInput("input_1", DimsCHW(3,224, 224),  UffInputOrder::kNCHW);
    // parser->registerOutput("net_vlad_layer_1/Reshape_1");
    // parser->registerOutput("conv_pw_5_relu/Relu6");
    // parser->registerOutput("block5_pool/MaxPool");
    // parser->registerOutput("fc1000/BiasAdd");
    // parser->registerOutput("predictions/Softmax"); //vgg16
    parser->registerOutput("Logits/Softmax");   //mobilenetv2
    // parser->registerOutput("reshape_2/Reshape");

    auto t_start = std::chrono::high_resolution_clock::now();
    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);

    // Save (Serialize) Engine to file
    IHostMemory* serMem = engine->serialize();
	  if( !serMem )
	  {
		    printf( "[ERROR] failed to serialize CUDA engine\n");
		    return false;
    }
    string engine_fname = fileName+".engine"; //"../uff_ready/tx2.engine";
    std::ofstream gieModelStream(engine_fname, ios::binary);
    if( !gieModelStream ) { cout << "[ERROR] cannot open file: " << engine_fname << endl; return 1; }
    cout << "Write Engine File to disk:" << engine_fname << endl;
    gieModelStream.write((const char*)serMem->data(), serMem->size());
    gieModelStream.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    cout << "loadModelAndCreateEngine Execution done in " << ms << " ms\n";


    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
    parser->destroy();


    // #endif
    #else
    // #if 1
    //
    // Load .engine file
    //
    // ICudaEngine * engine = loadCudaEngine( "../uff_ready/mobilenetv2_imagenet.uff.engine" );
    ICudaEngine * engine = loadCudaEngine( "../uff_ready/vgg16_imagenet.uff.engine" );
    #endif

    //TODO : Execute
    demo_exec( *engine );
    //demo_exec_async( *engine );

    cout << "Execution finished\n";

    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
