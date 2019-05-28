# A Standalone Example for TensorRT5 C++ API

## Files 
- data/lenet5.uff : A sample trained network. Input: "Input_0", output: "Binary_3". 
- [0-9].pgm: PGM files 
- common.h/cpp : Some needed utilities
- sampleUffMNIST.cpp : Borrowed from Nvidia samples. I feel this is overly complex
- bare_mnist.cpp : My simple adaptation. 

## Compile
```
g++ -std=c++11 bare_mnist.cpp common.cpp -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart -lnvinfer -lnvparsers
```

## Run 
```
sudo ./a.out
```

## Snippets 

### Create Engine
```c++
	auto fileName = string("data/lenet5.uff");
    int maxBatchSize = 1;
	auto parser = createUffParser();
    
    parser->registerInput("Input_0", DimsCHW(1, 28, 28));
    parser->registerOutput("Binary_3");
    
    // For defination of `loadModelAndCreateEngine` see bare_mnist.cpp. It can be copied as it is. 
    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);
```

### Synchronous Memory Transfer Usage 
This would be slightly inefficient because, the GPU does nothing as the memcpy happens. A better way is to use
CUDAstreams and memcpyasync.  

```c++
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
```


### Async Memory Transfer Usage
With cuda streams, memory transfers happen in DMA mode. So, execution can happen as the memory is loaded. 
Good for higher throughput. 

```c++
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
```
