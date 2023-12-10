#include <stdio.h>
#include "kernel.cu"
#include "declaration.h"

int main() {
    NET Net;
    BOOL Stop;
    REAL MinTestError;

    // Initialize network, random weights, and application
    InitializeRandoms();
    GenerateNetworkCUDA(&Net); // CUDA version
    RandomWeightsCUDA(&Net);   // CUDA version
    InitializeApplicationCUDA(&Net); // CUDA version

    Stop = FALSE;
    MinTestError = MAX_REAL;
    do {
        TrainNetCUDA(&Net, 10);  // CUDA version
        TestNetCUDA(&Net);       // CUDA version

        if (Net.TestError < MinTestError) {
            fprintf(f, " - saving Weights ...\n");
            MinTestError = Net.TestError;
            SaveWeightsCUDA(&Net);  // CUDA version
        }
        else if (Net.TestError > 1.2 * MinTestError) {
            fprintf(f, " - stopping Training and restoring Weights ...\n");
            Stop = TRUE;
            RestoreWeightsCUDA(&Net);  // CUDA version
        }
    } while (!Stop);

    TestNetCUDA(&Net);           // CUDA version
    EvaluateNetCUDA(&Net);       // CUDA version if applicable

    FinalizeApplicationCUDA(&Net); // CUDA version

    // Additional cleanup as needed, especially for GPU resources

    return 0;
}
/******************************************************************************
               A P P L I C A T I O N - S P E C I F I C   C O D E
 ******************************************************************************/

void NormalizeSunspotsCUDA(REAL *sunspots, REAL *normalized, int size) {
    REAL *d_sunspots, *d_min, *d_max;
    REAL h_min = MAX_REAL, h_max = MIN_REAL;
    
    // Allocate device memory
    cuda_ret =  cudaMalloc((void **)&d_sunspots, size * sizeof(REAL));
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMalloc((void **)&d_min, sizeof(REAL));
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret =  cudaMalloc((void **)&d_max, sizeof(REAL));
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");


    // Copy data to device
    cuda_ret = cudaMemcpy(d_sunspots, sunspots, size * sizeof(REAL), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemcpy(d_min, &h_min, sizeof(REAL), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemcpy(d_max, &h_max, sizeof(REAL), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    // Launch kernels
    int blockSize = 256;  // Adjust as necessary
    int numBlocks = (size + blockSize - 1) / blockSize;
    findMinKernel<<<numBlocks, blockSize>>>(d_sunspots, d_min, size);
    findMaxKernel<<<numBlocks, blockSize>>>(d_sunspots, d_max, size);

    // Copy back the results
    cudaMemcpy(&h_min, d_min, sizeof(REAL), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cudaMemcpy(&h_max, d_max, sizeof(REAL), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");


    // Normalize
    normalizeSunspotsKernel<<<numBlocks, blockSize>>>(d_sunspots, normalized, h_min, h_max, size);

    // Clean up
    cudaFree(d_sunspots);
    cudaFree(d_min);
    cudaFree(d_max);
}

void InitializeApplicationCUDA(NET* Net, REAL *d_sunspots, REAL *d_error) {
    INT Year, i;
    REAL h_error = 0.0;

    // Set network parameters
    Net->Alpha = 0.5;
    Net->Eta   = 0.05;
    Net->Gain  = 1;

    // Normalize sunspots (assuming this is already adapted for CUDA)
    NormalizeSunspotsCUDA(d_sunspots);

    // Allocate device memory for error
    cuda_ret = cudaMalloc((void **)&d_error, sizeof(REAL));
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy allocate device memory");

    cuda_ret = cudaMemcpy(d_error, &h_error, sizeof(REAL), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    // Launch kernel for TrainErrorPredictingMean
    int blockSize = 256; // Adjust as necessary
    int numBlocks = ((TRAIN_UPB - TRAIN_LWB + 1) * M + blockSize - 1) / blockSize;
    computeErrorKernel<<<numBlocks, blockSize>>>(d_sunspots, Mean, d_error, TRAIN_LWB, TRAIN_UPB, M, N);

    // Copy back and accumulate the results
    cuda_ret = cudaMemcpy(&h_error, d_error, sizeof(REAL), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    TrainErrorPredictingMean = h_error;

    // Reset error for next computation
    h_error = 0.0;
    cuda_ret = cudaMemcpy(d_error, &h_error, sizeof(REAL), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    // Launch kernel for TestErrorPredictingMean
    numBlocks = ((TEST_UPB - TEST_LWB + 1) * M + blockSize - 1) / blockSize;
    computeErrorKernel<<<numBlocks, blockSize>>>(d_sunspots, Mean, d_error, TEST_LWB, TEST_UPB, M, N);

    // Copy back and accumulate the results
    cuda_ret = cudaMemcpy(&h_error, d_error, sizeof(REAL), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    TestErrorPredictingMean = h_error;

    // Open file
    f = fopen("BPN.txt", "w");

    // Clean up
    cudaFree(d_error);
}
void FinalizeApplicationCUDA(NET* Net) {
    fclose(f);
}

void RandomWeightsCUDA(NET* Net) {
    INT l, i, j;
    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            int size = Units[l-1] + 1;
            RandomWeightsKernel<<<(size + 255) / 256, 256>>>(Net->Layer[l]->Weight[i], size);
            // Handle cudaDeviceSynchronize and error checking
        }
    }
}

/******************************************************************************
                          I N I T I A L I Z A T I O N
 ******************************************************************************/


void GenerateNetworkCUDA(NET* Net) {
    INT l, i;

    // Allocate memory for layers on the host
    Net->Layer = (LAYER**) malloc(NUM_LAYERS * sizeof(LAYER*));
   
    for (l = 0; l < NUM_LAYERS; l++) {
        // Allocate each layer on the host
        Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
        Net->Layer[l]->Units = Units[l];

        // Allocate Output and Error arrays for each layer on the GPU
	cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->Output), (Units[l] + 1) * sizeof(REAL));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
        cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->Error), (Units[l] + 1) * sizeof(REAL));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
        // Initialize the first element of Output array to BIAS
        if (l == 0) {
            cuda_ret = cudaMemcpy(Net->Layer[l]->Output, &BIAS, sizeof(REAL), cudaMemcpyHostToDevice);
            if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
         }

        // For layers other than the input layer, allocate Weight, WeightSave, and dWeight arrays
        if (l != 0) {
           cuda_ret =  cudaMalloc((void**)&(Net->Layer[l]->Weight), (Units[l] + 1) * sizeof(REAL*));
           if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
           cuda_ret =  cudaMalloc((void**)&(Net->Layer[l]->WeightSave), (Units[l] + 1) * sizeof(REAL*));
           if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
           cuda_ret =  cudaMalloc((void**)&(Net->Layer[l]->dWeight), (Units[l] + 1) * sizeof(REAL*));
           if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
            for (i = 1; i <= Units[l]; i++) {
                cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->Weight[i]), (Units[l-1] + 1) * sizeof(REAL));
                if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
                cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->WeightSave[i]), (Units[l-1] + 1) * sizeof(REAL));
                if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
                cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->dWeight[i]), (Units[l-1] + 1) * sizeof(REAL));
                if(cuda_ret != cudaSuccess) FATAL("Unable to allocate memory to the device");
            }
        }
    }

    // Initialize other network properties
    Net->InputLayer = Net->Layer[0];
    Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
    Net->Alpha = 0.9;
    Net->Eta = 0.25;
    Net->Gain = 1;
}

void SetInputCUDA(NET* Net, REAL* h_Input) {
    // Assuming Net->InputLayer->Output is already allocated on the GPU
    cudaMemcpy(Net->InputLayer->Output + 1, h_Input, Net->InputLayer->Units * sizeof(REAL), cudaMemcpyHostToDevice);

    // Handle errors for cudaMemcpy
}
void GetOutputCUDA(NET* Net, REAL* h_Output) {
    // Assuming Net->OutputLayer->Output is allocated on the GPU
    cudaMemcpy(h_Output, Net->OutputLayer->Output + 1, Net->OutputLayer->Units * sizeof(REAL), cudaMemcpyDeviceToHost);

    // Handle errors for cudaMemcpy
}

/******************************************************************************
            S U P P O R T   F O R   S T O P P E D   T R A I N I N G
 ******************************************************************************/
void SaveWeightsCUDA(NET* Net) {
    INT l, i;
    size_t size;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            size = (Net->Layer[l-1]->Units + 1) * sizeof(REAL);
            cudaMemcpy(Net->Layer[l]->WeightSave[i], Net->Layer[l]->Weight[i], size, cudaMemcpyDeviceToDevice);
            // Error checking for cudaMemcpy
        }
    }
}
void RestoreWeightsCUDA(NET* Net) {
    INT l, i;
    size_t size;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            size = (Net->Layer[l-1]->Units + 1) * sizeof(REAL);
            cudaMemcpy(Net->Layer[l]->Weight[i], Net->Layer[l]->WeightSave[i], size, cudaMemcpyDeviceToDevice);
            // Error checking for cudaMemcpy
        }
    }
}

/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/

void PropagateNetCUDA(NET* Net) {
    int blockSize = 256; // Adjust as necessary

    for (int l = 0; l < NUM_LAYERS - 1; l++) {
        int upperUnits = Net->Layer[l + 1]->Units;
        int numBlocks = (upperUnits + blockSize - 1) / blockSize;

        PropagateLayerKernel<<<numBlocks, blockSize>>>(
            Net->Layer[l]->Output,
            Net->Layer[l + 1]->Weight,
            Net->Layer[l + 1]->Output,
            Net->Layer[l]->Units,
            upperUnits,
            Net->Gain
        );

        // Handle cudaDeviceSynchronize and error checking
    }
}

/******************************************************************************
                  B A C K P R O P A G A T I N G   E R R O R S
 ******************************************************************************/

void ComputeOutputErrorCUDA(NET* Net, REAL* d_Target) {
    int blockSize = 256;
    int numBlocks = (Net->OutputLayer->Units + blockSize - 1) / blockSize;
    
    REAL h_NetError = 0.0;
    REAL* d_NetError;
    cudaMalloc((void**)&d_NetError, sizeof(REAL));
    cudaMemcpy(d_NetError, &h_NetError, sizeof(REAL), cudaMemcpyHostToDevice);

    ComputeOutputErrorKernel<<<numBlocks, blockSize>>>(
        Net->OutputLayer->Output, d_Target, Net->OutputLayer->Error, d_NetError, 
        Net->OutputLayer->Units, Net->Gain
    );

    // Handle cudaDeviceSynchronize and error checking

    cudaMemcpy(&h_NetError, d_NetError, sizeof(REAL), cudaMemcpyDeviceToHost);
    Net->Error = h_NetError;

    cudaFree(d_NetError);
}

void BackpropagateLayerCUDA(NET* Net, LAYER* Upper, LAYER* Lower) {
    int blockSize = 256; // Example block size, can be tuned
    int numBlocks = (Lower->Units + blockSize - 1) / blockSize;

    BackpropagateLayerKernel<<<numBlocks, blockSize>>>(
        Lower->Output, Lower->Error, Upper->Weight, Upper->Error, 
        Lower->Units, Upper->Units, Net->Gain
    );

    // Error checking and cudaDeviceSynchronize as needed
}

void BackpropagateNetCUDA(NET* Net) {
    for (int l = NUM_LAYERS - 1; l > 0; l--) {
        // Call the CUDA version of BackpropagateLayer
        BackpropagateLayerCUDA(Net, Net->Layer[l], Net->Layer[l-1]);

        // Synchronize after each layer's backpropagation
        // to ensure that the computation for one layer is complete
        // before starting the next layer
        cudaDeviceSynchronize();

        // Include error checking here
    }
}

void AdjustWeightsCUDA(NET* Net) {
    dim3 blockSize(16, 16); // Example block size, can be tuned
    for (int l = 1; l < NUM_LAYERS; l++) {
        int lowerUnits = Net->Layer[l-1]->Units;
        int upperUnits = Net->Layer[l]->Units;
        dim3 numBlocks((lowerUnits + blockSize.x - 1) / blockSize.x, 
                       (upperUnits + blockSize.y - 1) / blockSize.y);

        AdjustWeightsKernel<<<numBlocks, blockSize>>>(
            Net->Layer[l-1]->Output, Net->Layer[l]->Error, 
            Net->Layer[l]->Weight, Net->Layer[l]->dWeight, 
            lowerUnits, Net->Eta, Net->Alpha
        );

        // Error checking and cudaDeviceSynchronize as needed
    }
}
/******************************************************************************
                      S I M U L A T I N G   T H E   N E T
 ******************************************************************************/
void SimulateNetCUDA(NET* Net, REAL* d_Input, REAL* d_Output, REAL* d_Target, BOOL Training) {
    // Set the input in the input layer
    SetInputCUDA(Net, d_Input); // Assumes d_Input is a device pointer to the input data

    // Forward propagate the signals through the network
    PropagateNetCUDA(Net); // Implement this function to handle forward propagation in CUDA

    // Retrieve the output from the output layer
    GetOutputCUDA(Net, d_Output); // Assumes d_Output is a device pointer to store the output data

    // Compute the error at the output layer
    ComputeOutputErrorCUDA(Net, d_Target); // Assumes d_Target is a device pointer to the target data

    if (Training) {
        // Backpropagate the error through the network
        BackpropagateNetCUDA(Net); // Implement this function for backpropagation in CUDA

        // Adjust the weights based on the error
        AdjustWeightsCUDA(Net); // Implement this function to handle weight adjustment in CUDA
    }
}
void TrainNetCUDA(NET* Net, INT Epochs) {
    REAL *d_Input, *d_Output, *d_Target;
    REAL h_Output[M];

    // Allocate GPU memory for input, output, and target
    cudaMalloc((void**)&d_Input, N * sizeof(REAL));
    cudaMalloc((void**)&d_Output, M * sizeof(REAL));
    cudaMalloc((void**)&d_Target, M * sizeof(REAL));

    for (INT n = 0; n < Epochs * TRAIN_YEARS; n++) {
        INT Year = RandomEqualINT(TRAIN_LWB, TRAIN_UPB);

        // Copy input data to GPU
        cudaMemcpy(d_Input, &(Sunspots[Year - N]), N * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Target, &(Sunspots[Year]), M * sizeof(REAL), cudaMemcpyHostToDevice);

        // Run the simulation (forward and backward pass)
        SimulateNetCUDA(Net, d_Input, d_Output, d_Target, TRUE);

        // Optionally, retrieve the output from GPU for any CPU-side processing
        cudaMemcpy(h_Output, d_Output, M * sizeof(REAL), cudaMemcpyDeviceToHost);

        // You can do something with h_Output here if needed
    }

    // Free GPU memory
    cudaFree(d_Input);
    cudaFree(d_Output);
    cudaFree(d_Target);

    // Add error checking as needed
}

void TestNetCUDA(NET* Net) {
    REAL *d_Input, *d_Output, *d_Target;
    REAL h_Output[M];

    // Allocate GPU memory for input, output, and target
    cudaMalloc((void**)&d_Input, N * sizeof(REAL));
    cudaMalloc((void**)&d_Output, M * sizeof(REAL));
    cudaMalloc((void**)&d_Target, M * sizeof(REAL));

    TrainError = 0;
    for (INT Year = TRAIN_LWB; Year <= TRAIN_UPB; Year++) {
        cudaMemcpy(d_Input, &(Sunspots[Year - N]), N * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Target, &(Sunspots[Year]), M * sizeof(REAL), cudaMemcpyHostToDevice);

        SimulateNetCUDA(Net, d_Input, d_Output, d_Target, FALSE);
        TrainError += Net->Error; // Net->Error needs to be updated inside SimulateNetCUDA
    }

    TestError = 0;
    for (INT Year = TEST_LWB; Year <= TEST_UPB; Year++) {
        cudaMemcpy(d_Input, &(Sunspots[Year - N]), N * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Target, &(Sunspots[Year]), M * sizeof(REAL), cudaMemcpyHostToDevice);

        SimulateNetCUDA(Net, d_Input, d_Output, d_Target, FALSE);
        TestError += Net->Error; // Net->Error needs to be updated inside SimulateNetCUDA
    }

    // Print the normalized mean squared error
    fprintf(f, "\nNMSE is %0.3f on Training Set and %0.3f on Test Set",
             TrainError / TrainErrorPredictingMean,
             TestError / TestErrorPredictingMean);

    // Free GPU memory
    cudaFree(d_Input);
    cudaFree(d_Output);
    cudaFree(d_Target);

    // Add error checking as needed
}


void EvaluateNetCUDA(NET* Net) {
    REAL *d_Output, *d_Output_, *d_Sunspots, *d_Sunspots_;
    REAL h_Output[M], h_Output_[M];

    // Allocate memory on GPU
    cudaMalloc((void**)&d_Output, M * sizeof(REAL));
    cudaMalloc((void**)&d_Output_, M * sizeof(REAL));
    cudaMalloc((void**)&d_Sunspots, N * sizeof(REAL));
    cudaMalloc((void**)&d_Sunspots_, N * sizeof(REAL));

    fprintf(f, "\n\n\n");
    fprintf(f, "Year    Sunspots    Open-Loop Prediction    Closed-Loop Prediction\n\n");

    for (INT Year = EVAL_LWB; Year <= EVAL_UPB; Year++) {
        // Copy the required sunspot data to GPU
        cudaMemcpy(d_Sunspots, &(Sunspots[Year - N]), N * sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sunspots_, &(Sunspots_[Year - N]), N * sizeof(REAL), cudaMemcpyHostToDevice);

        // Run the simulation on GPU
        SimulateNetCUDA(Net, d_Sunspots, d_Output, &(Sunspots[Year]), FALSE);
        SimulateNetCUDA(Net, d_Sunspots_, d_Output_, &(Sunspots_[Year]), FALSE);

        // Copy the output back to host
        cudaMemcpy(h_Output, d_Output, M * sizeof(REAL), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Output_, d_Output_, M * sizeof(REAL), cudaMemcpyDeviceToHost);

        // Update Sunspots_ array on host
        Sunspots_[Year] = h_Output_[0];

        // Print results
        fprintf(f, "%d       %0.3f                   %0.3f                     %0.3f\n",
                FIRST_YEAR + Year,
                Sunspots[Year],
                h_Output[0],
                h_Output_[0]);
    }

    // Free GPU memory
    cudaFree(d_Output);
    cudaFree(d_Output_);
    cudaFree(d_Sunspots);
    cudaFree(d_Sunspots_);
}
/******************************************************************************
        R A N D O M S   D R A W N   F R O M   D I S T R I B U T I O N S
 ******************************************************************************/
void InitializeRandomsCUDA(curandState **state, int n) {
    cudaMalloc(state, n * sizeof(curandState));
    initRandomsKernel<<<(n + 255) / 256, 256>>>(*state, 4711);
    // Error checking
}

void RandomEqualINTCUDA(curandState *state, int *output, int low, int high, int n) {
    generateRandomIntsKernel<<<(n + 255) / 256, 256>>>(state, output, low, high, n);
    // Error checking
}

void RandomEqualREALCUDA(curandState *state, REAL *output, REAL low, REAL high, int n) {
    generateRandomRealsKernel<<<(n + 255) / 256, 256>>>(state, output, low, high, n);
    // Error checking
}
