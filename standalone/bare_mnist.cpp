
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
//-------------------------- END Utils ---------------------------//


void demo_exec( ICudaEngine& engine )
{

	float * input = new float[28*28*1]; 
	
	uint8_t * rawpgm = new uint8_t[28*28];
    readPGMFile( "data/0.pgm", rawpgm, 28, 28);
	for( int i=0; i<28*28; i++ ) 
		input[i]=1.0 - float(rawpgm[i])/255.; 
	float * output = new float[10]; 
	

	cout << "[demo_exec]Start\n"; 
	// Execution context 
    IExecutionContext* context = engine.createExecutionContext();
    
    // Bindings
    int nbinds = engine.getNbBindings(); 
    int inputIndex = engine.getBindingIndex( "Input_0" );
    int outputIndex = engine.getBindingIndex( "Binary_3" );
    cout << "nbinds = " << nbinds << endl;
    cout << "engine.getBindingIndex( \"Input_0\" ) ---> "<< inputIndex << endl; //0
    cout << "engine.getBindingIndex( \"Binary_3\" ) ---> "<< outputIndex << endl; //1
    
	// Create GPU buffers on device 
    void * buffers[2];
	CHECK( cudaMalloc( &buffers[inputIndex], 1*1*28*28*sizeof(float)) );
	CHECK( cudaMalloc( &buffers[outputIndex], 10*sizeof(float)) );

	
	// Host --> Device 
	cout << "cudaMemcpyHostToDevice\n" ;
	CHECK( cudaMemcpy( buffers[inputIndex], input, 1*28*28*sizeof(float), cudaMemcpyHostToDevice ) );
	
	// Execute 
	cout << "execute\n";
	            auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(1, &buffers[0]);
                auto t_end = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    cout << "Execution done in " << ms << " ms\n"; 
    
    // Device --> Host
	cout << "cudaMemcpyDeviceToHost\n";
	CHECK( cudaMemcpy( output, buffers[outputIndex], 10*sizeof(float), cudaMemcpyDeviceToHost ) );

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


void demo_exec_async( ICudaEngine& engine )
{

	float * input = new float[28*28*1]; 
	
	uint8_t * rawpgm = new uint8_t[28*28];
    readPGMFile( "data/0.pgm", rawpgm, 28, 28);
	for( int i=0; i<28*28; i++ ) 
		input[i]=1.0 - float(rawpgm[i])/255.; 
	float * output = new float[10]; 
	

	cout << "[demo_exec]Start\n"; 
	// Execution context 
    IExecutionContext* context = engine.createExecutionContext();
    
    // Bindings
    int nbinds = engine.getNbBindings(); 
    int inputIndex = engine.getBindingIndex( "Input_0" );
    int outputIndex = engine.getBindingIndex( "Binary_3" );
    cout << "nbinds = " << nbinds << endl;
    cout << "engine.getBindingIndex( \"Input_0\" ) ---> "<< inputIndex << endl; //0
    cout << "engine.getBindingIndex( \"Binary_3\" ) ---> "<< outputIndex << endl; //1
    
	// Create GPU buffers on device 
    void * buffers[2];
	CHECK( cudaMalloc( &buffers[inputIndex], 1*1*28*28*sizeof(float)) );
	CHECK( cudaMalloc( &buffers[outputIndex], 10*sizeof(float)) );

	
	#if 0
	// Host --> Device 
	cout << "cudaMemcpyHostToDevice\n" ;
	CHECK( cudaMemcpy( buffers[inputIndex], input, 1*28*28*sizeof(float), cudaMemcpyHostToDevice ) );
	
	// Execute 
	cout << "execute\n";
	            auto t_start = std::chrono::high_resolution_clock::now();
    context->execute(1, &buffers[0]);
                auto t_end = std::chrono::high_resolution_clock::now();
                float ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    cout << "Execution done in " << ms << " ms\n"; 
    
    // Device --> Host
	cout << "cudaMemcpyDeviceToHost\n";
	CHECK( cudaMemcpy( output, buffers[outputIndex], 10*sizeof(float), cudaMemcpyDeviceToHost ) );
	#endif 
	// Note: 
	// 	A better way is to use DMA with Async memory copy and cuda streams. See sampleMNISTAPI.cpp to know how to do it.	
	
	
	// Create Stream 
	cudaStream_t stream; 
	CHECK( cudaStreamCreate(&stream) ); 
	
	
	// DMA input batch data to device, do inference async, and DMA output back to host
	cout << "cudaMemcpyHostToDevice\n" ;
	CHECK( cudaMemcpyAsync( buffers[inputIndex], input, 1*28*28*sizeof(float), cudaMemcpyHostToDevice, stream ) );
	
	context->enqueue( 1, buffers, stream, nullptr ); 
	
	cout << "cudaMemcpyDeviceToHost\n";
	CHECK( cudaMemcpyAsync( output, buffers[outputIndex], 10*sizeof(float), cudaMemcpyDeviceToHost, stream ) );
	
	cudaStreamSynchronize( stream) ;
	 
		
	cout << "Output\n";
	for( int i=0 ; i<10; i++ ) {
		cout << i << ": " << output[i] << endl;
	}

	// Release 
	cout << "Release\n";
	cudaStreamDestroy( stream );
	CHECK( cudaFree(buffers[inputIndex]));
	CHECK( cudaFree(buffers[outputIndex]));
	delete [] input; 
	delete [] output;
}


int main()
{
    auto fileName = string("data/lenet5.uff"); //locateFile("data/lenet5.uff");
    std::cout << fileName << std::endl;

    int maxBatchSize = 1;
    auto parser = createUffParser();
    
    parser->registerInput("Input_0", DimsCHW(1, 28, 28), UffInputOrder::kNCHW);
    parser->registerOutput("Binary_3");
    
    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);
    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
    parser->destroy();
    
    //TODO : Execute
    //demo_exec( *engine );
    demo_exec_async( *engine );
    
    cout << "Execution finished\n"; 
    
    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
