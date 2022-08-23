
/* Julia_set_serial.cu
*  Created on: Mar 3, 2018
*      Julia set code by Abdallah Mohamed
*      Other files by EasyBMP (see BSD_(revised)_license.txt)
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "EasyBMP.h"

//Complex number definition
struct Complex {	// typedef is not required for C++
	float x; 		// real part is represented on x-axis in output image
	float y; 		// imaginary part is represented by y-axis in output image
};

//Function declarations
__global__ void compute_julia(uchar4*, int, int);
void save_image(uchar4*, const char*, int, int);
//__global__ void julia(uchar4*, float, float);
__device__ Complex add(Complex, Complex);
__device__ Complex mul(Complex, Complex);
__device__ float mag(Complex);

//main function
int main(void) {
	char* name = "test.bmp";

	int size = (3000 * 3000 * sizeof(uchar4));
	uchar4* pixel = (uchar4*)malloc(size);
	uchar4* d_p;
	int TILE_WIDTH = 32;
	int TILE_HEIGHT = 32;
	dim3 blocksize(TILE_WIDTH, TILE_HEIGHT);
	int w = 3000, h = 3000;
	int nblk_x = (w - 1) / TILE_WIDTH + 1;
	int nblk_y = (h - 1) / TILE_HEIGHT + 1;
	dim3 gridsize(nblk_x, nblk_y);

	cudaMalloc(&d_p, size);
	compute_julia << <gridsize, blocksize >> > (d_p,w, h);
	cudaMemcpy(pixel, d_p, size, cudaMemcpyDeviceToHost); \
	save_image(pixel, name, w, h);
	free(pixel);
	cudaFree(d_p);

	//compute_julia(name, 3000, 3000);	//width x height
	printf("Finished creating %s.\n", name);
	return 0;
}





// serial implementation of Julia set
__global__ void compute_julia(uchar4* pixels, int width, int height) {
	
	//PROBLEM SETTINGS (marked by '******')
	// **** Accuracy ****: lower values give less accuracy but faster performance
	int max_iterations = 400;
	int infinity = 20;													//used to check if z goes towards infinity

	// ***** Shape ****: other values produce different patterns. See https://en.wikipedia.org/wiki/Julia_set
	//Complex c = { 0.285, 0.01 }; 										//the constant in z = z^2 + c
	Complex c = { 0, -0.8 };
	// ***** Size ****: higher w means smaller size
	float w = 4;
	float h = w * height / width;										//preserve aspect ratio

	// LIMITS for each pixel
	float x_min = -w / 2, y_min = -h / 2;
	float x_incr = w / width, y_incr = h / height;
	
	//****************************************************
	//REQ: Parallelize the following for loop using CUDA 
	//****************************************************
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;


	if (x < width && y < height) {
		Complex z;
		z.x = x_min + x * x_incr;
		z.y = y_min + y * y_incr;

		int n = 0;
		do {
			z = add(mul(z, z), c);								// z = z^2 + c
		} while (mag(z) < infinity && n++ < max_iterations);	// keep looping until z->infinity or we reach max_iterations

		// color each pixel based on above loop
		if (n == max_iterations) {								// if we reach max_iterations before z reaches infinity, pixel is black 
			pixels[x + y * width] = { 0,0,0,0 };
		}
		else {												// if z reaches infinity, pixel color is based on how long it takes z to go to infinity
			unsigned char huer = (unsigned char)(200 * sqrt((float)n / max_iterations));
			unsigned char hueg = (unsigned char)(400 * sqrt((float)n / max_iterations));
			unsigned char hueb = (unsigned char)(4000* sqrt((float)n / max_iterations));
			pixels[x + y * width] = { huer,hueg,hueb,255 };
		}

	}
	
	
}

void save_image(uchar4* pixels, const char* filename, int width, int height) {
	BMP output;
	output.SetSize(width, height);
	output.SetBitDepth(24);
	// save each pixel to output image
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			uchar4 color = pixels[col + row * width];
			output(col, row)->Red = color.x;
			output(col, row)->Green = color.y;
			output(col, row)->Blue = color.z;
		}
	}
	output.WriteToFile(filename);
}



__device__ Complex add(Complex c1, Complex c2) {
	return{ c1.x + c2.x, c1.y + c2.y };
}

__device__ Complex mul(Complex c1, Complex c2) {
	return{ c1.x * c2.x - c1.y * c2.y, c1.x * c2.y + c2.x * c1.y };
}

__device__ float mag(Complex c) {
	return (float)sqrt((double)(c.x * c.x + c.y * c.y));
}