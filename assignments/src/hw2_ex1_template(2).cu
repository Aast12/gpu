#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

int main(int argc, char **argv) {
    int inputLength = atoi(argv[1]);  // Get input length from command line argument
    DataType *hostInput1, *hostInput2, *hostOutput, *resultRef;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    double hostToDeviceTime, deviceToHostTime;

    printf("The input length is %d\n", inputLength);

    // Allocate Host memory for input and output
    hostInput1 = (DataType *)malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType *)malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType *)malloc(inputLength * sizeof(DataType));
    resultRef = (DataType *)malloc(inputLength * sizeof(DataType));

    // Initialize hostInput1 and hostInput2 with random values
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % 1000;
        hostInput2[i] = rand() % 1000;
        resultRef[i] = hostInput1[i] + hostInput2[i];  // Reference result for validation
    }

    // Allocate GPU memory
    cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));

    // Copy data from Host to Device
    double startHostToDevice = getTime();
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    hostToDeviceTime = getTime() - startHostToDevice;
    printf("Data Copy Host to Device Time: %.6f seconds\n", hostToDeviceTime / 1.0e6);
    
    // Set grid and block dimensions
    int blockSize = 256;
    int gridSize = (inputLength + blockSize - 1) / blockSize;

    // Start timer for kernel execution
    double startKernel = getTime();

    // Launch the GPU kernel
    vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Stop timer for kernel execution
    double endKernel = getTime();
    printf("GPU Execution Time: %.6f seconds\n", (endKernel - startKernel) / 1.0e6);

    // Copy the result from Device to Host
    double startDeviceToHost = getTime();
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    deviceToHostTime = getTime() - startDeviceToHost;
    printf("Data Copy Device to Host Time: %.6f seconds\n", deviceToHostTime / 1.0e6);
    
    // Compare the result with the reference
    bool success = true;
    for (int i = 0; i < inputLength; i++) {
        if (abs(hostOutput[i] - resultRef[i]) > 1e-5) {
            success = false;
            break;
        }
    }

    if (success) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect!\n");
    }

    // Free GPU memory
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    // Free Host memory
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
