
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
*/

__global__ void initArray(double* d_b, int n) {
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    if (i < n) {
        d_b[i] = (double)i/n;
    }
}

int main()
{   
    double* a;
    const int n = 10000000;
    int num_bytes = n * sizeof(double);
    a = (double*) malloc(num_bytes);
    double start = clock();
    for (int i = 0; i < n; i++) {
        a[i] = (double)i/n;
    }
    double t = (clock() - start)/CLOCKS_PER_SEC;

    for (int i = 0; i < 5; i++) {
        printf("a[%d]: %.7f \n", i, a[i]);
    }

    for (int i = n-5; i < n; i++) {
        printf("a[%d]: %.7f \n", i, a[i]);
    }

    printf("\n Serial execution time: %.2f sec\n", t);
    free(a);


    int nThreads = 1024;
    int nBlocks = n / nThreads;
    if (n % 1024) nBlocks++;

    double* b = (double*)malloc(num_bytes);
    double* d_b;

    
    cudaMalloc(&d_b, num_bytes);
    //cudaMemset(d_b, 0, num_bytes);
    double x = clock();
    initArray<<<nBlocks, nThreads>>> (d_b,n);    
    cudaMemcpy(b, d_b, num_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    x = ((clock() - x) / CLOCKS_PER_SEC);
    cudaFree(d_b);

    for (int i = 0; i < 5; i++) {
        printf("b[%d]: %.7f \n", i, b[i]);
    }

    for (int i = n - 5; i < n; i++) {
        printf("b[%d]: %.7f \n", i, b[i]);
    }

    //printf("b[%d]: %.7f \n", 1023, b[1023]);
    //printf("b[%d]: %.7f \n", 1024, b[1024]);
    printf("\n Cuda execution time: %.2f sec\n", x);
    free(b);


}




/*
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/