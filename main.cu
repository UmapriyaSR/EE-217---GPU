#include<cuda_runtime.h>
#include <stdio.h>
#include "kernel.cu"
#include "declaration.h"
#include <math.h>
#define MIN_REAL      -HUGE_VAL
#define MAX_REAL      +HUGE_VAL

#define NUM_RNG 1 //has to be based on number of threads we can change this later
#define FATAL(msg) \
    do { \
        fprintf(stderr, "Error: %s\n", msg); \
        exit(EXIT_FAILURE); \
    } while (0)
const int blockSize = 256;
const int gridSize = (NUM_YEARS + blockSize - 1) / blockSize;


/******************************************************************************
               A P P L I C A T I O N - S P E C I F I C   C O D E
 ******************************************************************************/

void NormalizeSunspotsCUDA() {
    REAL *dev_sunspots, *dev_min, *dev_max, *dev_mean;
    REAL host_min=MIN_REAL, host_max=MIN_REAL, host_mean=0;
    // Allocate memory on the device
    cudaMalloc((void **)&dev_sunspots, NUM_YEARS * sizeof(REAL));
    cudaMalloc((void **)&dev_min, sizeof(REAL));
    cudaMalloc((void **)&dev_max, sizeof(REAL));
    cudaMalloc((void **)&dev_mean, sizeof(REAL));

    // Copy data from host to device
    cudaMemcpy(dev_sunspots, Sunspots, NUM_YEARS * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_min, &host_min, sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_max, &host_max, sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemset(dev_mean,0, sizeof(REAL));

    // Launch kernels (you need to define grid and block sizes)
    findMinKernel<<<gridSize, blockSize>>>(dev_sunspots, dev_min, NUM_YEARS);
    findMaxKernel<<<gridSize, blockSize>>>(dev_sunspots, dev_max, NUM_YEARS);
    normalizeSunspotsKernel<<<gridSize, blockSize>>>(dev_sunspots, host_min, host_max, dev_mean, NUM_YEARS, LO, HI);

    // Copy results back to host
    cudaMemcpy(&host_min, dev_min, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_max, dev_max, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(&Mean, dev_mean, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(Sunspots, dev_sunspots, NUM_YEARS * sizeof(REAL), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_sunspots);
    cudaFree(dev_min);
    cudaFree(dev_max);
    cudaFree(dev_mean);
}
void InitializeApplicationCUDA(NET* Net) {
    // Set network parameters
    Net->Alpha = 0.5;
    Net->Eta   = 0.05;
    Net->Gain  = 1;

    // Normalize sunspots (assuming NormalizeSunspots is also implemented in CUDA)
    NormalizeSunspotsCUDA();

    // Allocate memory for sunspots array on GPU
    REAL *dev_sunspots;
    cudaMalloc((void **)&dev_sunspots, NUM_YEARS * sizeof(REAL));
    cudaMemcpy(dev_sunspots, Sunspots, NUM_YEARS * sizeof(REAL), cudaMemcpyHostToDevice);

    // Calculate training and testing errors
    REAL *dev_trainError, *dev_testError;
    REAL trainError = 0, testError = 0;
    cudaMalloc((void **)&dev_trainError, sizeof(REAL));
    cudaMalloc((void **)&dev_testError, sizeof(REAL));
    cudaMemset(dev_trainError, 0, sizeof(REAL));
    cudaMemset(dev_testError, 0, sizeof(REAL));

    // Launch kernels for error calculations (grid and block sizes need to be set)
    computeErrorKernel<<<gridSize, blockSize>>>(dev_sunspots, Mean, dev_trainError, TRAIN_LWB, TRAIN_UPB, M);
    computeErrorKernel<<<gridSize, blockSize>>>(dev_sunspots, Mean, dev_testError, TEST_LWB, TEST_UPB, M);

    // Copy error results back to host
    cudaMemcpy(&trainError, dev_trainError, sizeof(REAL), cudaMemcpyDeviceToHost);
    cudaMemcpy(&testError, dev_testError, sizeof(REAL), cudaMemcpyDeviceToHost);

    // Update global variables
    TrainErrorPredictingMean = trainError;
    TestErrorPredictingMean = testError;

    // Free device memory
    cudaFree(dev_sunspots);
    cudaFree(dev_trainError);
    cudaFree(dev_testError);

    // File operations remain on the CPU
    f = fopen("BPN.txt", "w");
}
void FinalizeApplicationCUDA(NET* Net) {
    fclose(f);
}

void RandomWeightsCUDA(NET* Net) {
    unsigned long seed = 1234; // Example seed
    int threadsPerBlock = 256;
    int blocks;

    for (int l = 1; l < NUM_LAYERS; l++) {
        blocks = (Net->Layer[l]->Units * (Net->Layer[l-1]->Units + 1) + threadsPerBlock - 1) / threadsPerBlock;
        initializeRandomWeights<<<blocks, threadsPerBlock>>>(Net->Layer[l]->Weight, Net->Layer[l-1]->Units, Net->Layer[l]->Units, seed);
    }

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
/******************************************************************************
                          I N I T I A L I Z A T I O N
 ******************************************************************************/
void GenerateNetworkCUDA(NET* Net) {
    INT l, i;
    cudaError_t cuda_ret;

    // Allocate memory for layers on the host
    Net->Layer = (LAYER**) malloc(NUM_LAYERS * sizeof(LAYER*));

    for (l = 0; l < NUM_LAYERS; l++) {
        Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));
        Net->Layer[l]->Units = Units[l];

        // Allocate Output and Error arrays for each layer on the GPU
        cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->Output), (Units[l] + 1) * sizeof(REAL));
        if (cuda_ret != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device memory - Output\n");
            exit(EXIT_FAILURE);
        }

        cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->Error), (Units[l] + 1) * sizeof(REAL));
        if (cuda_ret != cudaSuccess) {
            fprintf(stderr, "Failed to allocate device memory - Error\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the first element of Output array to BIAS
        cuda_ret = cudaMemcpy(Net->Layer[l]->Output, &BIAS, sizeof(REAL), cudaMemcpyHostToDevice);
        if (cuda_ret != cudaSuccess) {
            fprintf(stderr, "Failed to copy data to device - Output[0]\n");
            exit(EXIT_FAILURE);
        }

        // Allocate Weight, WeightSave, and dWeight arrays
        if (l != 0) {
            for (i = 1; i <= Units[l]; i++) {
                cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->Weight[i]), (Units[l-1] + 1) * sizeof(REAL));
                cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->WeightSave[i]), (Units[l-1] + 1) * sizeof(REAL));
                cuda_ret = cudaMalloc((void**)&(Net->Layer[l]->dWeight[i]), (Units[l-1] + 1) * sizeof(REAL));

                // Add error checking for each cudaMalloc
                if (cuda_ret != cudaSuccess) {
                    fprintf(stderr, "Failed to allocate device memory - Weights\n");
                    exit(EXIT_FAILURE);
                }
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
    int threadsPerBlock = 256;
    int blocks;

    for (int l = 1; l < NUM_LAYERS; l++) {
        int numUnitsPrevLayer = Net->Layer[l-1]->Units;
        int numUnitsCurrLayer = Net->Layer[l]->Units;
 
        blocks = (numUnitsCurrLayer + threadsPerBlock - 1) / threadsPerBlock;

        saveWeightsKernel<<<blocks, threadsPerBlock>>>(Net->Layer[l]->Weight, 
            Net->Layer[l]->WeightSave, 
            numUnitsPrevLayer, 
            numUnitsCurrLayer);
    }

    // Error checking and synchronization
    cudaDeviceSynchronize();
    // ... (Error handling code)
}

void RestoreWeightsCUDA(NET* Net) {
    int threadsPerBlock = 256;
    int blocks;

    for (int l = 1; l < NUM_LAYERS; l++) {
        int numWeights = Net->Layer[l]->Units * (Net->Layer[l-1]->Units + 1);
        blocks = (numWeights + threadsPerBlock - 1) / threadsPerBlock;

        restoreWeightsKernel<<<blocks, threadsPerBlock>>>(Net->Layer[l]->Weight, Net->Layer[l]->WeightSave, numWeights);
    }

    // Error checking and synchronization
    cudaDeviceSynchronize();
    // ... (Error handling code)
}
/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/
void PropogateNetCUDA(NET* Net) {
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

        // Error checking and synchronization
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            // Handle the error appropriately
        }
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
        Net->OutputLayer->Output,
        Target,
        Net->OutputLayer->Error,
        Net->Gain,
        Net->OutputLayer->Units,
        d_NetError
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
    // Error checking and synchronization
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        // Handle the error appropriately
    }
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
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
            // Handle the error appropriately
        }
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


int main() {
    NET Net;
    BOOL Stop;
    REAL MinTestError;
    
    //cudaError thingie
    cudaError_t cuda_ret = cudaSuccess;
    
    //init cuda random states
    curandState* devStates;


    
    // Initialize network, random weights, and application
    InitializeRandomsCUDA(&devStates, NUM_RNG);
    GenerateNetworkCUDA(&Net); // CUDA version
    RandomWeightsCUDA(&Net);   // CUDA version
    InitializeApplicationCUDA(&Net); // CUDA version

    Stop = FALSE;
    MinTestError = MAX_REAL;
    do {
        TrainNetCUDA(&Net, 10);  // CUDA version
        TestNetCUDA(&Net);       // CUDA version

        if (TestError < MinTestError) {
            fprintf(f, " - saving Weights ...\n");
            MinTestError = TestError;
            SaveWeightsCUDA(&Net);  // CUDA version
        }
        else if (TestError > 1.2 * MinTestError) {
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
