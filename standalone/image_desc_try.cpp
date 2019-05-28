
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



int main()
{
    auto fileName = string("uff_ready/output_model.uff"); //locateFile("data/lenet5.uff");
    std::cout << fileName << std::endl;

    int maxBatchSize = 1;
    auto parser = createUffParser();
    
    parser->registerInput("input_1", DimsCHW(3, 480, 752));
    parser->registerOutput("resulting0");
    
    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);
    if (!engine)
        RETURN_AND_LOG(EXIT_FAILURE, ERROR, "Model load failed");
    parser->destroy();
    
    //TODO : Execute
    //demo_exec( *engine );
    //demo_exec_async( *engine );
    
    cout << "Execution finished\n"; 
    
    engine->destroy();
    shutdownProtobufLibrary();
    return EXIT_SUCCESS;
}
