#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void addVec(int* C, int* A, int* B, int size){
	while(size>0){
		*C = *A+*B;
		*C++;
		*A++;
		*B++;
		size--;
	}
}

int main(void) {
	double start = clock();
	int n = 50000000;
	int* A = (int*) calloc(n, sizeof(int));
	int* B = (int*) calloc(n, sizeof(int));
	int* C = (int*) calloc(n, sizeof(int));
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
	addVec(C,A,B,n);
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
 * Execution time: 0.30 sec
 *
 * int n = 5000000000000;
 * Not enough memory
 */
