
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CHK(call){cudaError_t err = call; if(err != cudaSuccess){printf("Error%d: %s:%d\n",err,__FILE__,__LINE__);printf(cudaGetErrorString(err));cudaDeviceReset();cudaDeviceReset();}}

__global__ void cudaSumOne(float* d_a, float* d_blockSum,int size) {
    int ix = threadIdx.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float partialSum[1024];

    if (i < size) { 
        partialSum[ix] = d_a[i];
        __syncthreads();

        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (ix % (2 * stride) == 0)
                partialSum[ix] += partialSum[ix + stride];
            __syncthreads();
        }
        if (ix == 0)
            d_blockSum[blockIdx.x] = partialSum[0];
    }
}

__global__ void cudaSumTwo(float* d_a, float* d_blockSum, int size) {
    int ix = threadIdx.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float partialSum[1024];

    
    if (i < size) {
        partialSum[ix] = d_a[i];
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
            if (ix < stride)
                partialSum[ix] += partialSum[ix + stride];
            __syncthreads();
        }
        if (ix == 0)
            d_blockSum[blockIdx.x] = partialSum[0];
    }
}



__global__ void cudaSumThree(float* d_a, float* d_blockSum, int size) {
    int ix = threadIdx.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    float* idata = d_a + blockIdx.x * blockDim.x;
    __syncthreads();
    if (i < size) {
        for (int stride = 1; stride < blockDim.x; stride *= 2) {
            if (ix % (2 * stride) == 0)
                idata[ix] += idata[ix + stride];
            __syncthreads();
        }
        if (ix == 0)
            d_blockSum[blockIdx.x] = idata[0];

    }
}



__global__ void cudaSumFour(float* d_a, float* d_blockSum, int size) {
    int ix = threadIdx.x;
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    
    float* idata = d_a + blockIdx.x * blockDim.x;
    __syncthreads();
    if (i < size) {        
        for (int stride = blockDim.x >> 1; stride >= 1; stride >>= 1) {
            if (ix < stride)
                idata[ix] += idata[ix + stride];                                
            __syncthreads();
        }
        if (ix == 0) {
            d_blockSum[blockIdx.x] = idata[0];
        }
        
    }
}


int main()
{
    int n = pow(2, 24);
    int nThreads = 1024;
    int nBlocks = n / nThreads;
    if (n % 1024) nBlocks++;

    
    float* a = (float*) malloc(n * sizeof(float));
    float* blockSum = (float*)malloc(nBlocks * sizeof(float));
    float* d_a; float* d_blockSum; 

    for (int i = 0; i < n; i++) {
        a[i] = (rand() % 256);
    }

    
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    
   

    ////////////////////////////////////////Q1
    CHK(cudaMalloc(&d_a, n * sizeof(float)));
    CHK(cudaMalloc(&d_blockSum, nBlocks * sizeof(float)));
    CHK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));
    double t = clock();
    cudaSumOne << <nBlocks, nThreads >> > (d_a, d_blockSum,n);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    CHK(cudaMemcpy(blockSum, d_blockSum, nBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float parSum = 0;
    for (int i = 0; i < nBlocks; i++) {
        parSum += blockSum[i];
    }

    t = (double)(clock() - t)*1000 / CLOCKS_PER_SEC;

    printf("Reducing and array of %d floats on a grid of (%d,1,1) blocks, each block with (%d,%d,%d) threads\n\n", n, nBlocks, nThreads, 1, 1);


    printf("Using shared memory, More divergence: GPU time: %06.2f ms      GPU sum: %.2f\n", t, parSum);
    



    //////////////////////////////////////Q2
    CHK(cudaFree(d_blockSum));
    CHK(cudaMalloc(&d_blockSum, nBlocks * sizeof(float)));
    t = clock();
    cudaSumTwo << <nBlocks, nThreads >> > (d_a, d_blockSum,n);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    CHK(cudaMemcpy(blockSum, d_blockSum, nBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    parSum = 0;
    for (int i = 0; i < nBlocks; i++) {
        parSum += blockSum[i];
    }
    t = (clock() - t)*1000/ CLOCKS_PER_SEC;
    printf("Using shared memory, Less divergence: GPU time: %06.2f ms      GPU sum: %.2f\n", t, parSum);





    //////////////////////////////////////Q3
    CHK(cudaFree(d_blockSum));
    CHK(cudaMalloc(&d_blockSum, nBlocks * sizeof(float)));
    


    t = clock();
    cudaSumThree << <nBlocks, nThreads >> > (d_a, d_blockSum,n);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    CHK(cudaMemcpy(blockSum, d_blockSum, nBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    parSum = 0;
    for (int i = 0; i < nBlocks; i++) {
        parSum += blockSum[i];
    }
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    printf("Using Global memory, More divergence: GPU time: %06.2f ms      GPU sum: %.2f\n", t, parSum);

    CHK(cudaFree(d_a));
    CHK(cudaMalloc(&d_a, n * sizeof(float)));
    CHK(cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice));



    /////////////////////////////////////Q4
    cudaFree(d_blockSum);
    cudaMalloc(&d_blockSum, nBlocks * sizeof(float));    

    t = clock();
    cudaSumFour<< <nBlocks, nThreads >> > (d_a, d_blockSum,n);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    CHK(cudaMemcpy(blockSum, d_blockSum, nBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    parSum = 0;
    for (int i = 0; i < nBlocks; i++) {
        parSum += blockSum[i];
    }
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    printf("Using Global memory, Less divergence: GPU time: %06.2f ms      GPU sum: %.2f\n", t, parSum);


    //printf("Sum %f\n", sum);
    free(a);
    cudaFree(d_blockSum);
    cudaFree(d_a);
    return 0;
}


