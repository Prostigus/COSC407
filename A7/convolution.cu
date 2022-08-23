#include "EasyBMP.h"
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>	// for uchar4 struct
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MIN(x,y) (  (y) ^ (((x) ^ (y)) & -((x) < (y))) )
#define MAX(x,y) (  (x) ^ (((x) ^ (y)) & -((x) < (y))) )

#define CHK(call){cudaError_t err = call; if(err != cudaSuccess){printf("Error%d: %s:%d\n",err,__FILE__,__LINE__);printf(cudaGetErrorString(err));cudaDeviceReset();cudaDeviceReset();}}

//****************************************************************************************************************
// PARALLEL FUNCTIONS
//****************************************************************************************************************
	/*
	TODO: 	Provide CUDA implementation for parallelizing the two SERIAL functions: convolution_8bits and convolution_32Bits
			Make sure to check for errors from CUDA API calls and from Kernel Launch. 
			Also, time your parallel code and compute the speed-up.
	*/


__global__ void convolution_8bits(const unsigned char* const image_in, unsigned char* const image_out, const int height, const int width, const float *filter, const int filter_width){
	//only filters with width = odd_number are allowed
	if (filter_width % 2 == 0) {
		return;
	}

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {

		float sum = 0.0f;

		for (int row_f = -filter_width / 2; row_f <= filter_width / 2; ++row_f)
			for (int col_f = -filter_width / 2; col_f <= filter_width / 2; ++col_f) {
				int row_i = MIN(MAX(y + row_f, 0), (height - 1));
				int col_i = MIN(MAX(x + col_f, 0), (width - 1));
				float pxl_image = image_in[row_i * width + col_i];
				float pxl_filter = filter[(row_f + filter_width / 2) * filter_width + col_f + filter_width / 2];
				sum += pxl_image * pxl_filter;
			}
		image_out[y * width + x] = sum;
	}
}

//	This function applies the convolution kernel (denoted by filter) to every pixel of the input image (image_in)
//	Constraints:- Both image_in and image_out are in RGBA format (32-bit pixels as uchar4)
//				- Filter is a square matrix (float) and its width is odd number. The sum of all its values is 1 (normalized)


__global__ void convolution_32bits( const uchar4* const image_in, uchar4 * const image_out, int height, int width,  float*  const filter, const int filter_width, unsigned char* R_in, unsigned char* G_in, unsigned char* B_in, unsigned char* A_in){
	//break the input image (uchar4 matrix) into 4 channels (four char matrices): Red, Green, Blue, and Alpha

	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x<( width * height)) {	//break each pixel in input image
		uchar4 pxl = image_in[x];
		R_in[x] = pxl.x;
		G_in[x] = pxl.y;
		B_in[x] = pxl.z;
		A_in[x] = pxl.w;
	}	
}

__global__ void combine(const unsigned char* const R_out, unsigned char* G_out,const  unsigned char* const B_out,const unsigned char* const A_out, uchar4* const image_out,int height, int width) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < (height * width))
		image_out[x] = make_uchar4(R_out[x], G_out[x], B_out[x], A_out[x]);
}

//**************************************************************
//No need to parallelize any of the functions below this comment
//**************************************************************

//This function reads a BMP image using the EasyBMP library and returns a 1D array representing the RGBA values of the image pixels
//image_out->x is Red, image_out->y is Green, image_out->z is Blue, image_out->w is Alpha
//how to use:	1- in the calling function, declare these variables:	uchar4* img = NULL;	int width = 0, height = 0;
//				2- then call this function								readBMP(filename, &img, &width, &height); 
void readBMP(const char* FileName, uchar4 **image_out, int* width, int* height){
	BMP img;
	img.ReadFromFile(FileName);
	*width = img.TellWidth();
	*height = img.TellHeight();
	uchar4 *const img_uchar4 = (uchar4*)malloc(*width * *height * sizeof(int));
	// save each pixel to image_out as uchar4 in row-major format
	for (int row = 0; row <*height; row++)
		for (int col = 0; col < *width; col++)
			img_uchar4[col + row * *width] = make_uchar4(img(col, row)->Red, img(col, row)->Green, img(col, row)->Blue, img(col, row)->Alpha);	//use row-major
	*image_out = img_uchar4;
}

//This function writes a BMP image using the EasyBMP library
//how to use: in the calling function, call		writeBMP(destination_filename, source_image_array, width, height); 
void writeBMP(const char* FileName, uchar4 *image, int width, int height){
	BMP output;
	output.SetSize(width, height);
	output.SetBitDepth(24);
	// save each pixel to the output image
	for (int row = 0; row < height; row++){		//for each row
		for (int col = 0; col <  width; col++){	//for each col
			uchar4 rgba = image[col + row * width];
			output(col, row)->Red = rgba.x;
			output(col, row)->Green = rgba.y;
			output(col, row)->Blue = rgba.z;
			output(col, row)->Alpha = rgba.w;
		}
	}
	output.WriteToFile(FileName);

}

//Normalize image filter (sum of all values should be 1) 
// the filter is a 2D float array
void normalizeFilter(float* filter, int width){
	//find the sum
	float sum = 0;
	for (int i = 0; i < width*width; i++)
		sum += filter[i];
	//normalize
	for (int i = 0; i < width*width; i++)
		filter[i] /= sum;
}

//this Function reads the convolution-filter image 
//Contrasting: Filter is 32 bit RGPA image. The filter must be sqaure. Filter width must be an odd number 
float* readFilter(const char* filter_image_name, int* filter_width){
	int filterHeight;	//for testing that height = width
	//read filter image as 32 bit RGPA bitmap and check the constraints (square, odd width)
	uchar4* filterImageUchar;
	readBMP(filter_image_name, &filterImageUchar, filter_width, &filterHeight);
	if (*filter_width != filterHeight || *filter_width % 2 == 0){
		fprintf(stderr, "Non-square filters or filters with even width are not supported yet. Program terminated!\n");
		exit(1);
	}
	//convert every pixel to a float number representing its grayscale intensity. Formula used is 0.21 R + 0.72 G + 0.07 B
	float* filter = (float*)malloc(*filter_width * *filter_width * sizeof(float));
	for (int i = 0; i < *filter_width * *filter_width; i++){
		uchar4 element = filterImageUchar[i];
		filter[i] = 0.21 * element.x + 0.72 * element.y + 0.07 * element.z; 
	}
	//Normalization makes sure that the sum of all values in the filter is 1 
	normalizeFilter(filter, *filter_width);
	//return result
	return filter;
}

/*

void serial(){
	int filter_width;
	const char* filter_image_name = "filter_blur_21.bmp";	//filter width = 21 pixels
	const char* image_in_name = "okanagan.bmp";
	const char* image_out_name = "okanagan_blur.bmp";

	//load filter
	float* filter = readFilter(filter_image_name, &filter_width);
	printf("Filter loaded...\n");

	//load input image
	int width, height;
	uchar4* image_in;
	readBMP(image_in_name, &image_in, &width, &height);	//image_in will have all pixel information, each pixel as uchar4
	printf("Input image loaded...\n");

	//apply convolution filter to input image
	uchar4* image_out = (uchar4*)malloc(width*height*sizeof(uchar4));	//reserve space in the memory for the output image
	printf("Applying the convolution filter...\n");
	int t = clock();
	convolution_32bits(image_in, image_out, height, width, filter, filter_width);	//filter applied to image_in, results saved in image_out
	t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
	printf("Convolution filter applied. Time taken: %d.%d seconds\n", t / 1000, t % 1000);
	
	//save results to output image
	writeBMP(image_out_name, image_out, width, height);
	printf("Output image saved.\nProgram finished!\n");
}*/
void parallel(){
	//launch your cuda kernel from here
	int filter_width;
	const char* filter_image_name = "filter_blur_21.bmp";	//filter width = 21 pixels
	const char* image_in_name = "okanagan.bmp";
	const char* image_out_name = "okanagan_blur.bmp";

	//load filter
	float* filter = readFilter(filter_image_name, &filter_width);
	printf("Filter loaded...\n");


	//load input image
	int width, height;
	uchar4* image_in;
	readBMP(image_in_name, &image_in, &width, &height);	//image_in will have all pixel information, each pixel as uchar4
	printf("Input image loaded...\n");

	uchar4* image_out = (uchar4*)malloc(width * height * sizeof(uchar4));	//reserve space in the memory for the output image


	int nThreads = 1024;
	int nBlocks = (height * width) / nThreads;
	if ((height * width) % 1024) nBlocks++;

	int sz = width * height * sizeof(unsigned char);

	
	unsigned char* d_R_in; unsigned char* d_G_in; unsigned char* d_B_in; unsigned char* d_A_in;
	CHK(cudaMalloc(&d_R_in, sz));
	CHK(cudaMalloc(&d_G_in, sz));
	CHK(cudaMalloc(&d_B_in, sz));
	CHK(cudaMalloc(&d_A_in, sz));


	unsigned char* d_R_out; unsigned char* d_G_out; unsigned char* d_B_out; unsigned char* d_A_out;
	CHK(cudaMalloc(&d_R_out, sz));
	CHK(cudaMalloc(&d_G_out, sz));
	CHK(cudaMalloc(&d_B_out, sz));
	CHK(cudaMalloc(&d_A_out, sz));


	float* d_filter;
	CHK(cudaMalloc(&d_filter, (filter_width * filter_width * sizeof(float))));
	CHK(cudaMemcpy(d_filter, filter, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice));

	uchar4* d_image_in;
	CHK(cudaMalloc(&d_image_in, width * height * sizeof(uchar4)));
	CHK(cudaMemcpy(d_image_in, image_in, width * height * sizeof(uchar4), cudaMemcpyHostToDevice));

	uchar4* d_image_out;
	CHK(cudaMalloc(&d_image_out, width * height * sizeof(uchar4)));


	printf("Applying the convolution filter...\n");

	int t = clock();
	convolution_32bits<<<nBlocks,nThreads>>>(d_image_in, d_image_out, height, width, d_filter, filter_width,d_R_in,d_G_in,d_B_in,d_A_in);	
	CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
	

	dim3 blockSize(32, 32, 1);
	int nblk_x = (width - 1) / 32+1;
	int nblk_y = (height - 1) / 32+1;
	dim3 gridSize(nblk_x, nblk_y);
	convolution_8bits << <gridSize, blockSize >> > (d_R_in, d_R_out, height, width, d_filter, filter_width);
	convolution_8bits << <gridSize, blockSize >> > (d_G_in, d_G_out, height, width, d_filter, filter_width);
	convolution_8bits << <gridSize, blockSize >> > (d_B_in, d_B_out, height, width, d_filter, filter_width);
	convolution_8bits << <gridSize, blockSize >> > (d_A_in, d_A_out, height, width, d_filter, filter_width);
	CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
	
	combine << <nBlocks, nThreads >> > (d_R_out,d_G_out, d_B_out, d_A_out, d_image_out, height, width);
	CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());

	CHK(cudaMemcpy(image_out, d_image_out, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
	CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
	
	t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
	printf("Convolution filter applied. Time taken: %d.%d seconds\n", t / 1000, t % 1000);

	writeBMP(image_out_name, image_out, width, height);
	printf("Output image saved.\nProgram finished!\n");

	free(image_in);
	free(image_out);
	cudaFree(d_image_in);
	cudaFree(d_image_out);
	cudaFree(d_R_in);
	cudaFree(d_G_in);
	cudaFree(d_B_in);
	cudaFree(d_A_in);
	cudaFree(d_R_out);
	cudaFree(d_G_out);
	cudaFree(d_B_out);
	cudaFree(d_A_out);
	cudaFree(d_filter);

}
//MAIN: testing convolution with a blur filter
int main(){
	//serial();
	parallel();
}



// serial time 8.803 seconds
// parallel time 0.353 seconds