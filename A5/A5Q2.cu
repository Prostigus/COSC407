
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


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




