#include <stdio.h>
#include <sys/time.h>
#include <iostream>

#define DataType double

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

//@@ Insert code to implement timer start

double getTime() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time.tv_sec * 1.0e6 + time.tv_usec;
}

//@@ Insert code to implement timer stop

void getElapsedTime(char *msg, double start) {
    double stop = getTime();
    printf("%s Time: %.6f seconds\n", msg, (stop - start) / 1.0e6);
}

int main(int argc, char **argv) {
    int inputLength;
    DataType *hostInput1;
    DataType *hostInput2;
    DataType *hostOutput;
    DataType *resultRef;
    DataType *deviceInput1;
    DataType *deviceInput2;
    DataType *deviceOutput;

    //@@ Insert code below to read in inputLength from args

    inputLength = atoi(argv[1]);

    printf("The input length is %d\n", inputLength);

    //@@ Insert code below to allocate Host memory for input and output

    hostInput1 = (DataType *) malloc(inputLength * sizeof(DataType));
    hostInput2 = (DataType *) malloc(inputLength * sizeof(DataType));
    hostOutput = (DataType *) malloc(inputLength * sizeof(DataType));
    resultRef = (DataType *) malloc(inputLength * sizeof(DataType));

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() % 1000;
        hostInput2[i] = rand() % 1000;
        resultRef[i] = hostInput1[i] + hostInput2[i]; // Reference result for validation
    }

    //@@ Insert code below to allocate GPU memory here

    cudaMalloc((void **) &deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc((void **) &deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc((void **) &deviceOutput, inputLength * sizeof(DataType));

    //@@ Insert code to below to Copy memory to the GPU here

    double start = getTime();
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    getElapsedTime("Data Copy Host to Device", start);

    //@@ Initialize the 1D grid and block dimensions here

    int blockSize = 256;
    int gridSize = (inputLength + blockSize - 1) / blockSize;

    //@@ Launch the GPU Kernel here

    start = getTime();
    vecAdd<<<gridSize, blockSize>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();
    getElapsedTime("GPU Execution", start);

    //@@ Copy the GPU memory back to the CPU here

    start = getTime();
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    getElapsedTime("Data Copy Device to Host", start);

    //@@ Insert code below to compare the output with the reference

    bool success = true;
    for (int i = 0; i < inputLength; i++) {
        auto delta = abs(hostOutput[i] - resultRef[i]);
        if (delta > 1e-5) {
            success = false;
            std::cout << "Failed delta: " << delta << std::endl;
            break;
        }
    }

    if (success) {
        printf("Results are correct!\n");
    } else {
        printf("Results are incorrect!\n");
    }

    //@@ Free the GPU memory here

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);
    free(resultRef);

    return 0;
}
