#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType float

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns) {
        DataType value = 0.0;
        for (int k = 0; k < numAColumns; k++) {
            value += A[row * numAColumns + k] * B[k * numBColumns + col];
        }
        C[row * numBColumns + col] = value;
    }
}

double getTime() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1.0e6 + time.tv_usec;
}

int main(int argc, char **argv) {
    // Reading matrix dimensions from command line arguments
    int numARows = atoi(argv[1]);
    int numAColumns = atoi(argv[2]);
    int numBRows = numAColumns;  // numBRows = numAColumns for valid matrix multiplication
    int numBColumns = atoi(argv[3]);
    int numCRows = numARows;
    int numCColumns = numBColumns;

    DataType *hostA, *hostB, *hostC, *resultRef;
    DataType *deviceA, *deviceB, *deviceC;

    printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n",
           numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

    // Allocate Host memory for input and output
    hostA = (DataType *)malloc(numARows * numAColumns * sizeof(DataType));
    hostB = (DataType *)malloc(numBRows * numBColumns * sizeof(DataType));
    hostC = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));
    resultRef = (DataType *)malloc(numCRows * numCColumns * sizeof(DataType));

    // Initialize hostA and hostB with random numbers, and compute reference result on CPU
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numAColumns; j++) {
            hostA[i * numAColumns + j] = rand() % 10;
        }
    }
    for (int i = 0; i < numBRows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            hostB[i * numBColumns + j] = rand() % 10;
        }
    }
    // Compute reference result
    for (int i = 0; i < numARows; i++) {
        for (int j = 0; j < numBColumns; j++) {
            resultRef[i * numBColumns + j] = 0.0;
            for (int k = 0; k < numAColumns; k++) {
                resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
            }
        }
    }

    // Allocate GPU memory
    cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(DataType));
    cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(DataType));

    // Copy data from Host to Device
    double startHostToDevice = getTime();
    cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
    double hostToDeviceTime = getTime() - startHostToDevice;
    printf("Data Copy Host to Device Time: %.6f seconds\n", hostToDeviceTime / 1.0e6);

    // Set grid and block dimensions
    int blockSize = 16; // Assuming square blocks
    dim3 block(blockSize, blockSize);
    dim3 grid((numCColumns + blockSize - 1) / blockSize, (numCRows + blockSize - 1) / blockSize);

    // Start GPU kernel timer
    double startKernelTime = getTime();

    // Launch the GPU Kernel
    gemm<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Stop GPU kernel timer
    double endKernelTime = getTime();
    printf("GPU Execution Time: %.6f seconds\n", (endKernelTime - startKernelTime) / 1.0e6);

    // Copy the result from Device to Host
    double startDeviceToHost = getTime();
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);
    double deviceToHostTime = getTime() - startDeviceToHost;
    printf("Data Copy Device to Host Time: %.6f seconds\n", deviceToHostTime / 1.0e6);

    // Compare the result with the reference
    bool success = true;
    for (int i = 0; i < numCRows; i++) {
        for (int j = 0; j < numCColumns; j++) {
            if (abs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j]) > 1e-5) {
                success = false;
                break;
            }
        }
    }

    if (success) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect!\n");
    }

    // Free GPU memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    // Free Host memory
    free(hostA);
    free(hostB);
    free(hostC);
    free(resultRef);

    return 0;
}
