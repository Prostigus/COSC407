

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
	double start = clock();
	int n = 1000000000;
	int* A = (int*) malloc(n * sizeof(int));
	int* B = (int*) malloc(n * sizeof(int));
	int* C = (int*) malloc(n * sizeof(int));
	if(A == NULL){
		fprintf(stderr,"Not enough memory");
		exit(EXIT_FAILURE);
	}
	if(B == NULL){
			fprintf(stderr,"Not enough memory");
			exit(EXIT_FAILURE);
		}
	if(C == NULL){
			fprintf(stderr,"Not enough memory");
			exit(EXIT_FAILURE);
		}
	int sum = 0;
	for(int i = 0; i<n;i++){
		A[i] = i*3;
		B[i] = -i*3;
		C[i] = A[i] + B[i];
		sum = sum + C[i];
	}
	free(A);
	free(B);
	double end = clock();
	double difference = (end - start)/CLOCKS_PER_SEC;
	printf("Sum: %d\n",sum);
	printf("Execution time: %.2f sec",difference);
	free(C);
	return EXIT_SUCCESS;
}

/*
 * 1: Successful
 * Sum: 0
 * Execution time: 0.00 sec
 *
 * 10: Successful
 * Sum: 0
 * Execution time: 0.00 sec
 *
 * 50: Successful
 * Sum: 0
 * Execution time: 0.00 sec
 *
 * 100000000: Successful
 * Sum: 0
 * Execution time: 0.71 sec
 *
 * 1000000000: Failure
 * not enough memory
 */
