#include "declaration.h"
#define BLOCK_SIZE 256
 
#include <curand.h>
#include <curand_kernel.h>

/******************************************************************************
        R A N D O M S   D R A W N   F R O M   D I S T R I B U T I O N S
 ******************************************************************************/
__global__ void initRandomsKernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void generateRandomIntsKernel(curandState *state, int *output, int low, int high, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        output[id] = low + (int)(curand_uniform(&state[id]) * (high - low + 1));
    }
}

__global__ void generateRandomRealsKernel(curandState *state, REAL *output, REAL low, REAL high, int n) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n) {
        output[id] = low + curand_uniform(&state[id]) * (high - low);
    }
}

/******************************************************************************
               A P P L I C A T I O N - S P E C I F I C   C O D E
 ******************************************************************************/
__global__ void findMinKernel(REAL *sunspots, REAL *min, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicMin(min, sunspots[i]);
    }
}

__global__ void findMaxKernel(REAL *sunspots, REAL *max, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicMax(max, sunspots[i]);
    }
}
__global__ void normalizeSunspotsKernel(REAL *sunspots, REAL *normalized, REAL min, REAL max, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        normalized[i] = (sunspots[i] - min) / (max - min) * (HI - LO) + LO;
    }
}


__global__ void computeErrorKernel(REAL *sunspots, REAL mean, REAL *error, int startYear, int endYear, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + startYear < endYear) {
        REAL out = sunspots[i + startYear + N]; // Adjust index as needed
        REAL err = mean - out;
        atomicAdd(error, 0.5 * sqr(err));
    }
}
/******************************************************************************
                          I N I T I A L I Z A T I O N
 ******************************************************************************/

__global__ void RandomWeightsKernel(REAL* weights, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        // Assuming RandomEqualREAL_GPU is a device function for generating random numbers
        weights[i] = RandomEqualREAL_GPU(-0.5, 0.5);
    }
}

/******************************************************************************
                     P R O P A G A T I N G   S I G N A L S
 ******************************************************************************/
__global__ void PropagateLayerKernel(REAL *lowerOutput, REAL **upperWeight, REAL *upperOutput, int lowerUnits, int upperUnits, REAL gain) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= upperUnits) {
        REAL sum = 0;
        for (int j = 0; j <= lowerUnits; j++) {
            sum += upperWeight[i][j] * lowerOutput[j];
        }
        upperOutput[i] = 1 / (1 + exp(-gain * sum));
    }
}
/******************************************************************************
                  B A C K P R O P A G A T I N G   E R R O R S
 ******************************************************************************/
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
__global__ void AdjustWeightsKernel(REAL *lowerOutput, REAL *error, REAL **weight, REAL **dWeight, int lowerUnits, REAL eta, REAL alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i <= lowerUnits && j < gridDim.x) { // Assuming gridDim.x = upperUnits
        REAL out = lowerOutput[i];
        REAL err = error[j];
        REAL dWeightVal = dWeight[j][i];

        // Update the weight and the delta
        weight[j][i] += eta * err * out + alpha * dWeightVal;
        dWeight[j][i] = eta * err * out;
    }
}

/******************************************************************************
                      S I M U L A T I N G   T H E   N E T
 ******************************************************************************/
