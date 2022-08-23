#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int* addVec2(int* A, int* B, int size){
	int* temp = (int*) calloc(size, sizeof(int));
	if(temp == NULL){
		return NULL;
	}
	while(size>0){
		*temp = *A + *B;
		*temp++;
		*A++;
		*B++;
		size--;
	}
	return temp;
}

int main(void) {
	double start = clock();
	int n = 50000000;
	int* A = (int*) calloc(n, sizeof(int));
	int* B = (int*) calloc(n, sizeof(int));
	if(A == NULL){
		fprintf(stderr,"Not enough memory");
		exit(EXIT_FAILURE);
	}
	if(B == NULL){
		fprintf(stderr,"Not enough memory");
		exit(EXIT_FAILURE);
	}
	int* C = addVec2(A,B,n);
	if(C == NULL){
		fprintf(stderr,"Not enough memory");
		exit(EXIT_FAILURE);
	}
	free(A);
	free(B);
	for(int i=0; i<10;i++){
		printf("%d ",C[i]);
	}
	double end = clock();
	double difference = (end - start)/CLOCKS_PER_SEC;
	printf("\nExecution time: %.2f sec",difference);
	free(C);
	return EXIT_SUCCESS;
}



/* int n = 50000000;
 * 0 0 0 0 0 0 0 0 0 0
 * Execution time: 0.29 sec
 *
 * int n = 5000000000000;
 * Not enough memory
 */
