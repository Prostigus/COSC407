/**************************************************************

 The program reads a BMP image file and creates a new
 image that is the negative or desaturated of the input file.

 **************************************************************/

#include "qdbmp.h"
#include <stdio.h>
#include <omp.h>

typedef enum {desaturate, negative} ImgProcessing ;

/* Creates a negative image of the input bitmap file */
int main() {
	const char* inFile = "okanagan.bmp";
	const char* outFile = "okanagan_processed.bmp";
	const ImgProcessing processingType = desaturate; //or negative


	UCHAR r, g, b;
	UINT width, height;
	UINT x, y;
	BMP* bmp;

	/* Read an image file */
	bmp = BMP_ReadFile(inFile);
	BMP_CHECK_ERROR(stdout, -1);

	/* Get image's dimensions */
	width = BMP_GetWidth(bmp);
	height = BMP_GetHeight(bmp);

	double t = omp_get_wtime();
	//edit to change # of threads
	int num_thread = 16;

#pragma omp parallel num_threads(num_thread) private(x,y,r,g,b)
	{
		int my_id = omp_get_thread_num();
		int my_n = width/num_thread;
		int start = my_id * my_n;
		int end = start + my_n;



	/* Iterate through all the image's pixels */
	for (x = start; x < end; ++x) {
		for (y = 0; y < height; ++y) {
			/* Get pixel's RGB values */
			BMP_GetPixelRGB(bmp, x, y, &r, &g, &b);

			/* Write new RGB values */
			if(processingType == negative)
				BMP_SetPixelRGB(bmp, x, y, 255 - r, 255 - g, 255 - b);
			else if(processingType == desaturate){
				UCHAR gray = r * 0.3 + g * 0.59 + b * 0.11;
				BMP_SetPixelRGB(bmp, x, y, gray, gray, gray);
			}
		}
	}

	}
	/* calculate and print processing time*/
	t = 1000 * (omp_get_wtime() - t);
	printf("Finished image processing in %.1f ms.", t);

	/* Save result */
	BMP_WriteFile(bmp, outFile);
	BMP_CHECK_ERROR(stdout, -2);

	/* Free all memory allocated for the image */
	BMP_Free(bmp);

	return 0;
}



/*
 * Serial: 299.0 ms
 * 2 threads: 237.0 ms
 * 4 threads: 202.0 ms
 * 8 threads: 186.0 ms
 * 16 threads: 178.0 ms
 */
