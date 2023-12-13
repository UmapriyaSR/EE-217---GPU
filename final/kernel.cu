#include "declaration.h"
#define BLOCK_SIZE 256
#include <curand.h>
#include <curand_kernel.h>
#include<iostream>
#include <math.h>

/******************************************************************************
        R A N D O M S   D R A W N   F R O M   D I S T R I B U T I O N S
 ******************************************************************************/
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}





__global__ void initRandomsKernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void generateRandomIntsKernel(curandState *state, int *output, int low, int high, int num) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < num) {
        output[id] = low + (int)(curand_uniform(&state[id]) * (high - low + 1));
    }
}

__global__ void generateRandomRealsKernel(curandState *state, REAL *output, REAL low, REAL high, int num) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < num) {
        output[id] = low + curand_uniform(&state[id]) * (high - low);
    }
}



/*__device__ float RandomEqualREAL_GPU(float min, float max)
{
    curandState state;
    curand_init(1234, threadIdx.x, 0, &state);  // Adjust the seed (1234 in this example)
    
    // Generate a random value between min and max
    float randomValue = curand_uniform(&state) * (max - min) + min;
    
    return randomValue;
} //temp


/******************************************************************************
               A P P L I C A T I O N - S P E C I F I C   C O D E
 ******************************************************************************/

__global__ void findMaxKernel(REAL *sunspots, REAL *max,int numyears) {
    extern __shared__ REAL sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared memory
    sdata[tid] = (i < numyears) ? sunspots[i] : MIN_REAL;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] > sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) max[blockIdx.x] = sdata[0];
}

__global__ void findMinKernel(REAL *sunspots, REAL *min,int numyears) {
    extern __shared__ REAL sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared memory
    sdata[tid] = (i < numyears) ? sunspots[i] : MAX_REAL;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = (sdata[tid] < sdata[tid + s]) ? sdata[tid] : sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) min[blockIdx.x] = sdata[0];
}

__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL min, REAL max, REAL *mean, int numYears, REAL lo, REAL hi) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < numYears) {
        sunspots[index] = ((sunspots[index] - min) / (max - min)) * (hi - lo) + lo;
        atomicAdd(mean, sunspots[index] / numYears); // Use atomicAdd for concurrent addition
    }
}
__global__ void computeErrorKernel(REAL *sunspots, REAL mean, REAL *error, int startYear, int endYear,int m) {
   int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index + startYear <= endYear) {
        for (int i = 0; i < m; i++) {
            REAL out = sunspots[index + startYear + i];
            REAL err = mean - out;
            atomicAdd(error, 0.5 * err * err); // Atomic add to avoid race conditions
        }
    }
}
/******************************************************************************
                          I N I T I A L I Z A T I O N
 ******************************************************************************/
__global__ void initializeRandomWeights(REAL** weight, int numUnitsPrevLayer, int numUnitsCurrLayer, unsigned long seed) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numUnitsCurrLayer && j < (numUnitsPrevLayer + 1)) {
        int idx = i * (numUnitsPrevLayer + 1) + j;

        curandState state;
        curand_init(seed, idx, 0, &state);

        weight[i][j] = curand_uniform(&state) - 0.5; // Generates values between -0.5 and 0.5
    }
}
/******************************************************i************************
            S U P P O R T   F O R   S T O P P E D   T R A I N I N G
 ******************************************************************************/

__global__ void saveWeightsKernel(REAL** weight, REAL** weightSave, int numUnitsPrevLayer, int numUnitsCurrLayer) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numUnitsCurrLayer && j < (numUnitsPrevLayer + 1)) {
        //int idx = i * (numUnitsPrevLayer + 1) + j;
        weightSave[i][j] = weight[i][j];
    }
}


__global__ void restoreWeightsKernel(REAL** weight, REAL** weightSave, int numRows, int numCols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (i < numRows && j < numCols) {
        weight[i][j] = weightSave[i][j];
    }
}
/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/

/*
__global__ void backpropagateLayerKernel(REAL* lowerOutput, REAL* upperWeight, REAL* upperError, REAL* lowerError, int lowerUnits, int upperUnits, REAL gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lowerUnits) {
        REAL out = lowerOutput[i + 1]; // Assuming output array starts at index 1
        REAL err = 0;
        for (int j = 1; j <= upperUnits; j++) {
            err += upperWeight[j * (lowerUnits + 1) + i] * upperError[j];
        }
        lowerError[i + 1] = gain * out * (1 - out) * err;
    }
}*/

__device__ REAL sigmoid(REAL x, REAL gain) {
    return 1.0 / (1.0 + exp(-gain * x));
}

__global__ void propagateNetKernel(NET* net) {
    INT l = blockIdx.x * blockDim.x + threadIdx.x;

    if (l < NUM_LAYERS - 1) {
        LAYER* lower = net->Layer[l];
        LAYER* upper = net->Layer[l + 1];

        INT i = blockIdx.x * blockDim.x + threadIdx.x + 1;

        if (i <= upper->Units) {
            REAL sum = 0;
            for (INT j = 0; j <= lower->Units; j+= 2) {
                //loop unrolling optimization
                sum += upper->Weight[i][j] * lower->Output[j];
                sum += upper->Weight[i][j+1] * lower->Output[j+1];
            }
            upper->Output[i] = sigmoid(sum, net->Gain);
        }
    }
}

/******************************************************************************
                  B A C K P R O P A G A T I N G   E R R O R S
 ******************************************************************************/

/*
__global__ void ComputeOutputErrorKernel(REAL *output, REAL *target, REAL *error, int units, REAL gain, REAL *netError) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < units) {
        REAL out = output[i];
        REAL err = target[i] - out;  // Assuming zero-based indexing for target
        error[i] = gain * out * (1 - out) * err;
        atomicAdd(netError, 0.5 * err * err); // Accumulate the squared error
    }
}

__global__ void BackpropagateLayerKernel(REAL *lowerOutput, REAL *lowerError, REAL **upperWeight, REAL *upperError, int lowerUnits, int upperUnits, REAL gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < lowerUnits) {
        REAL out = lowerOutput[i];
        REAL err = 0;
        for (int j = 0; j < upperUnits; j++) {
            err += upperWeight[j][i] * upperError[j];
        }
        lowerError[i] = gain * out * (1 - out) * err;
    }
}

__global__ void adjustWeightsKernel(REAL* lowerOutput, REAL* upperError, REAL* weight, REAL* dWeight, int lowerUnits, int upperUnits, REAL eta, REAL alpha) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // For upper units
    int j = blockIdx.x * blockDim.x + threadIdx.x; // For lower units

    if (i <= upperUnits && j <= lowerUnits) {
        REAL out = lowerOutput[j];
        REAL err = upperError[i];
        REAL dw = dWeight[i * (lowerUnits + 1) + j];
        weight[i * (lowerUnits + 1) + j] += eta * err * out + alpha * dw;
        dWeight[i * (lowerUnits + 1) + j] = eta * err * out;
    }
}


/******************************************************************************
                      S I M U L A T I N G   T H E   N E T
 ******************************************************************************/
