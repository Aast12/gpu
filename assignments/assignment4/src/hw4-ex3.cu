#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#define gpuCheck(stmt)                                               \
  do {                                                               \
      cudaError_t err = stmt;                                        \
      if (err != cudaSuccess) {                                      \
          printf("ERROR. Failed to run stmt %s\n", #stmt);           \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuBLAS status
#define cublasCheck(stmt)                                            \
  do {                                                               \
      cublasStatus_t err = stmt;                                     \
      if (err != CUBLAS_STATUS_SUCCESS) {                            \
          printf("ERROR. Failed to run cuBLAS stmt %s\n", #stmt);    \
          break;                                                     \
      }                                                              \
  } while (0)

// Macro to check the cuSPARSE status
#define cusparseCheck(stmt)                                          \
  do {                                                               \
      cusparseStatus_t err = stmt;                                   \
      if (err != CUSPARSE_STATUS_SUCCESS) {                          \
          printf("ERROR. Failed to run cuSPARSE stmt %s\n", #stmt);  \
          break;                                                     \
      }                                                              \
  } while (0)


struct timeval t_start, t_end;

void cputimer_start() {
    gettimeofday(&t_start, 0);
}

void cputimer_stop(const char* info) {
    gettimeofday(&t_end, 0);
    double time = (1000000.0 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec - t_start.tv_usec);
    printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time);
}

double cputimer_stop_ret() {
    gettimeofday(&t_end, 0);
    double time = (1e6 * (t_end.tv_sec - t_start.tv_sec) + t_end.tv_usec - t_start.tv_usec);
    return time * 1e-6;
}

// Initialize the sparse matrix needed for the heat time step
void matrixInit(double* A, int* ArowPtr, int* AcolIndx, int dimX,
                double alpha) {
    // Stencil from the finete difference discretization of the equation
    double stencil[] = {1, -2, 1};
    // Variable holding the position to insert a new element
    size_t ptr = 0;
    // Insert a row of zeros at the beginning of the matrix
    ArowPtr[1] = ptr;
    // Fill the non zero entries of the matrix
    for (int i = 1; i < (dimX - 1); ++i) {
        // Insert the elements: A[i][i-1], A[i][i], A[i][i+1]
        for (int k = 0; k < 3; ++k) {
            // Set the value for A[i][i+k-1]
            A[ptr] = stencil[k];
            // Set the column index for A[i][i+k-1]
            AcolIndx[ptr++] = i + k - 1;
        }
        // Set the number of newly added elements
        ArowPtr[i + 1] = ptr;
    }
    // Insert a row of zeros at the end of the matrix
    ArowPtr[dimX] = ptr;
}

int main(int argc, char** argv) {
    int device = 0; // Device to be used
    int dimX; // Dimension of the metal rod
    int nsteps; // Number of time steps to perform
    double alpha = 0.4; // Diffusion coefficient
    double* temp; // Array to store the final time step
    double* A; // Sparse matrix A values in the CSR format
    int* ARowPtr; // Sparse matrix A row pointers in the CSR format
    int* AColIndx; // Sparse matrix A col values in the CSR format
    int nzv; // Number of non zero values in the sparse matrix
    double* tmp; // Temporal array of dimX for computations
    size_t bufferSize = 0; // Buffer size needed by some routines
    void* buffer = nullptr; // Buffer used by some routines in the libraries
    int concurrentAccessQ; // Check if concurrent access flag is set
    double zero = 0; // Zero constant
    double one = 1; // One constant
    double norm; // Variable for norm values
    double error; // Variable for storing the relative error
    double tempLeft = 200.; // Left heat source applied to the rod
    double tempRight = 300.; // Right heat source applied to the rod
    cublasHandle_t cublasHandle; // cuBLAS handle
    cusparseHandle_t cusparseHandle; // cuSPARSE handle

    // Read the arguments from the command line
    dimX = atoi(argv[1]);
    nsteps = atoi(argv[2]);

    // Print input arguments
    printf("The X dimension of the grid is %d \n", dimX);
    printf("The number of time steps to perform is %d \n", nsteps);

    // Get if the cudaDevAttrConcurrentManagedAccess flag is set
    gpuCheck(cudaDeviceGetAttribute(&concurrentAccessQ, cudaDevAttrConcurrentManagedAccess, device));

    // Calculate the number of non zero values in the sparse matrix. This number
    // is known from the structure of the sparse matrix
    nzv = 3 * dimX - 6;

    //@@ Insert the code to allocate the temp, tmp and the sparse matrix
    //@@ arrays using Unified Memory
    cputimer_start();

    gpuCheck(cudaMallocManaged(&temp, sizeof(double) * dimX));
    gpuCheck(cudaMallocManaged(&tmp, sizeof(double) * dimX));
    gpuCheck(cudaMallocManaged(&A, sizeof(double) * nzv));
    gpuCheck(cudaMallocManaged(&ARowPtr, sizeof(int) * (dimX + 1)));
    gpuCheck(cudaMallocManaged(&AColIndx, sizeof(int) * nzv));

    cputimer_stop("Allocating device memory");

    // Check if concurrentAccessQ is non zero in order to prefetch memory
    if (concurrentAccessQ) {
        cputimer_start();
        //@@ Insert code to prefetch in Unified Memory asynchronously to CP
        gpuCheck(cudaMemPrefetchAsync(temp, sizeof(double) * dimX, cudaCpuDeviceId));
        gpuCheck(cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, cudaCpuDeviceId));
        gpuCheck(cudaMemPrefetchAsync(A, sizeof(double) * nzv, cudaCpuDeviceId));
        gpuCheck(cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1), cudaCpuDeviceId));
        gpuCheck(cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, cudaCpuDeviceId));
        cputimer_stop("Prefetching GPU memory to the host");
    }

    // Initialize the sparse matrix
    cputimer_start();
    matrixInit(A, ARowPtr, AColIndx, dimX, alpha);
    cputimer_stop("Initializing the sparse matrix on the host");

    //Initiliaze the boundary conditions for the heat equation
    cputimer_start();
    memset(temp, 0, sizeof(double) * dimX);
    temp[0] = tempLeft;
    temp[dimX - 1] = tempRight;
    cputimer_stop("Initializing memory on the host");

    if (concurrentAccessQ) {
        cputimer_start();
        //@@ Insert code to prefetch in Unified Memory asynchronously to the GPU
        gpuCheck(cudaMemPrefetchAsync(temp, sizeof(double) * dimX, device));
        gpuCheck(cudaMemPrefetchAsync(tmp, sizeof(double) * dimX, device));
        gpuCheck(cudaMemPrefetchAsync(A, sizeof(double) * nzv, device));
        gpuCheck(cudaMemPrefetchAsync(ARowPtr, sizeof(int) * (dimX + 1), device));
        gpuCheck(cudaMemPrefetchAsync(AColIndx, sizeof(int) * nzv, device));
        cputimer_stop("Prefetching GPU memory to the device");
    }

    //@@ Insert code to create the cuBLAS handle
    cublasCheck(cublasCreate(&cublasHandle));

    //@@ Insert code to create the cuSPARSE handle
    cusparseCheck(cusparseCreate(&cusparseHandle));

    //@@ Insert code to set the cuBLAS pointer mode to CUSPARSE_POINTER_MODE_HOST
    cublasCheck(cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST));

    //@@ Insert code to call cusparse api to create the mat descriptor used by cuSPARSE
    cusparseSpMatDescr_t matA;
    cusparseCheck(cusparseCreateCsr(
        &matA,
        dimX,
        dimX,
        nzv,
        ARowPtr,
        AColIndx,
        A,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    ));
    cusparseDnVecDescr_t vTemp;
    cusparseCheck(cusparseCreateDnVec(
        &vTemp,
        dimX,
        temp,
        CUDA_R_64F
    ));
    cusparseDnVecDescr_t vTmp;
    cusparseCheck(cusparseCreateDnVec(
        &vTmp,
        dimX,
        tmp,
        CUDA_R_64F
    ));


    //@@ Insert code to call cusparse api to get the buffer size needed by the sparse matrix per
    //@@ vector (SMPV) CSR routine of cuSPARSE
    cusparseCheck(cusparseSpMV_bufferSize(
        cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one,
        matA,
        vTemp,
        &zero,
        vTmp,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &bufferSize
    ));
    //@@ Insert code to allocate the buffer needed by cuSPARSE
    gpuCheck(cudaMalloc(&buffer, bufferSize));

    double spmvTime = 0.0;
    int it = 0;
    // Perform the time step iterations
    for (it = 0; it < nsteps; ++it) {
        //@@ Insert code to call cusparse api to compute the SMPV (sparse matrix multiplication) for
        //@@ the CSR matrix using cuSPARSE. This calculation corresponds to:
        //@@ tmp = 1 * A * temp + 0 * tmp
        cputimer_start();
        cusparseSpMV(
            cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one,
            matA,
            vTemp,
            &zero,
            vTmp,
            CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            buffer
        );
        double elapsed = cputimer_stop_ret();
        spmvTime += elapsed;


        //@@ Insert code to call cublas api to compute the axpy routine using cuBLAS.
        //@@ This calculation corresponds to: temp = alpha * tmp + temp;
        cublasCheck(
            cublasDaxpy(
                cublasHandle,
                dimX,
                &alpha,
                tmp,
                1,
                temp,
                1
            )
        );
        //@@ Insert code to call cublas api to compute the norm of the vector using cuBLAS
        //@@ This calculation corresponds to: ||temp||
        cublasCheck(
            cublasDnrm2(
                cublasHandle,
                dimX,
                temp,
                1,
                &norm
            )
        );

        // If the norm of A*temp is smaller than 10^-4 exit the loop
        if (norm < 1e-4)
            break;
    }

    // Calculate the exact solution using thrust
    thrust::device_ptr<double> thrustPtr(tmp);
    thrust::sequence(thrustPtr, thrustPtr + dimX, tempLeft,
                     (tempRight - tempLeft) / (dimX - 1));

    // Calculate the relative approximation error:
    one = -1;
    //@@ Insert the code to call cublas api to compute the difference between the exact solution
    //@@ and the approximation
    //@@ This calculation corresponds to: tmp = -temp + tmp
    cublasCheck(
        cublasDaxpy(
            cublasHandle,
            dimX,
            &one,
            temp,
            1,
            tmp,
            1
        )
    );

    //@@ Insert the code to call cublas api to compute the norm of the absolute error
    //@@ This calculation corresponds to: || tmp ||
    cublasCheck(
        cublasDnrm2(
            cublasHandle,
            dimX,
            tmp,
            1,
            &norm
        )
    );

    error = norm;
    //@@ Insert the code to call cublas api to compute the norm of temp
    //@@ This calculation corresponds to: || temp ||
    cublasCheck(
        cublasDnrm2(
            cublasHandle,
            dimX,
            temp,
            1,
            &norm
        )
    );

    // Calculate the relative error
    error = error / norm;
    printf("The relative error of the approximation is %f\n", error);

    // Flops approximation
    // Each matrix multiplication has 2 * nzv flops, one multiplication and one addition
    // per non-zero value in the matrix
    double totalFlops = it * 2 * nzv;
    double gflops = (totalFlops / spmvTime) / 1e9;

    printf("Iterations: %d\n", it);
    printf("Total Operations: %f\n", totalFlops);
    printf("SMPV time : %f\n", spmvTime);
    printf("GFLOPs: %f\n", gflops);

    //@@ Insert the code to destroy the mat descriptor
    cusparseCheck(cusparseDestroySpMat(matA));
    cusparseCheck(cusparseDestroyDnVec(vTemp));
    cusparseCheck(cusparseDestroyDnVec(vTmp));

    //@@ Insert the code to destroy the cuSPARSE handle
    cusparseCheck(cusparseDestroy(cusparseHandle));

    //@@ Insert the code to destroy the cuBLAS handle
    cublasCheck(cublasDestroy(cublasHandle));

    //@@ Insert the code for deallocating memory
    if (buffer)
        gpuCheck(cudaFree(buffer));
    gpuCheck(cudaFree(temp));
    gpuCheck(cudaFree(tmp));
    gpuCheck(cudaFree(A));
    gpuCheck(cudaFree(ARowPtr));
    gpuCheck(cudaFree(AColIndx));


    return 0;
}
