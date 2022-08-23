#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int* vecCreate(int size){
	int* A = malloc(size * sizeof(int));
	if(A == NULL){
		return NULL;
	}
	for(int i=0; i<size; i++){
		A[i] = i;
	}
	return A;
}


int* vecCreateOpenMP(int size, int num_thread){
	if(size%num_thread != 0){
		fprintf(stderr,"Error: number of threads must be divisible by vector size.\n");
		exit(EXIT_FAILURE);
		return(NULL);
	}
	int* A = malloc(size * sizeof(int));
	#pragma omp parallel num_threads(num_thread)
	{
		int my_id = omp_get_thread_num();
		int my_n = size/num_thread;
		int start = my_id * my_n;
		int end = start + my_n -1;
		for(int i=start; i<=end; i++){
					A[i] = i;
		}
	}
	return A;
}

int main(int argc, char *argv[]) {
	int n = 50000000;
	double start = clock();
	int* A = vecCreate(n);
	if(A == NULL){
		fprintf(stderr,"Not enough memory.\n");
		exit(EXIT_FAILURE);
	}
	double end = clock();
	double difference = (end - start)/CLOCKS_PER_SEC;
	printf("Using serial code\n");
	printf("v[%d] = %d",n-1,A[n-1]);
	printf("\nTime: %.2f sec\n",difference);
	free(A);
	start = clock();
	int* B = vecCreateOpenMP(n,4);
	if(B == NULL){
		fprintf(stderr,"Not enough memory");
		exit(EXIT_FAILURE);
	}
	end = clock();
	difference = (end - start)/CLOCKS_PER_SEC;
	printf("Using parallel code\n");
	printf("v[%d] = %d",n-1,B[n-1]);
	printf("\nTime: %.2f sec\n",difference);
	free(B);
	return 0;

}

/*
 * Using serial code
 * v[49999999] = 49999999
 * Time: 0.11 sec
 *
 *Using parallel code
 *v[49999999] = 49999999
 *Time: 0.04 sec
 */
