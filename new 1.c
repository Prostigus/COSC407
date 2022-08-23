#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>

//define a struct, point (remember that a struct is like an object in java but without methods)
typedef struct {
   float x, y;
} point;  

//function returns Euclidean distance between two points (this to
device float distance(point p1, point p2) {
   return (float)sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

__global__ pointDistance(int numPoints,point d_points,int d_indexOfClosest){
	// Start: parallelize this part to GPU – make changes in other parts as needed
   // Put results in indexOfClosest array (e.g. indexOfClosest[0] = 5 means closest point to p0 is p5)
   int x = blockIdx.x * blockDim.x + ThreadIdx.x;   
   __shared__ points[numPoints];
   if(x<numPoints)
	   points[x] = d_points[x];
	__syncthreads();
   
   
   float minDist = FLT_MAX; // max float value 
   if(x < numPoints){
	   for (int other = 0; other < numPoints; other++){
			if (x == other) return;
			float dist = distance(points[x], points[other]);
			if (dist < minDist) {
				 minDist = dist;
				 d_indexOfClosest[x] = other;
			} 
		}
   }
}

void main(){
   const int numPoints = 20000;

   // Create and initialize array of points (random for the purpose of this question)
   point points[numPoints];
   for (int i = 0; i < numPoints; i++) {      //Don't run this loop on the GPU
       points[i].x = rand() % 20000 - 10000;  //x range [-10,000,10,000]
       points[i].y = rand() % 20000 - 10000;  //y range [-10,000,10,000]
   }

   // Create array to hold results.
   int indexOfClosest[numPoints];
  
   
   
   //kernel
   int nThreads = 1024;
   int nBlocks = numPoints/nThreads;
   if(numPoints%1024) nBlocks++;
   
   
	point d_points;
	int d_indexOfClosest;	
	cudaMalloc(&d_points, numPoints *sizeof(point));
	cudaMalloc(&d_indexOfClosest, numPoints *sizeof(int));
	cudaMemcpy(d_points, points, numPoints *sizeof(point), cudaMemcpyHostToDevice);
	<<<nBlocks,nThreads>>> (numPoints, d_points, d_indexOfClosest);
	cudaMemcpy(indexOfClosest, d_indexOfClosest, numPoints *sizeof(int), cudaMemcpyDeviceToHost);
   

   // Print results for first 10 points
   for (int i = 0; i < 10; i++)              
       printf("%d -> %d\n", i, indexOfClosest[i]);
}//end main