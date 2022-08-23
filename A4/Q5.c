/*
 ============================================================================
 Name        : A4.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello OpenMP World in C
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define NRA 20
#define NCA 30
#define NCB 10
/**
 * Hello OpenMP World prints the number of threads and the current thread id
 */

void count_sort(int a[], int n) {
	int i, j, count;
	int* temp = malloc(n * sizeof(int));
#pragma omp for private(j,count)
	for (i = 0; i < n; i++){
		//count all elements < a[i]
		count = 0;
		for (j = 0; j < n; j++)
			if(a[j]<a[i] ||(a[j]==a[i] && j<i))
				count++;
		//place a[i] at right order
		temp[count] = a[i];
	}
	memcpy(a, temp, n * sizeof(int));
	free(temp);
}
int main (int argc, char *argv[]) {
	int n = 10000;
	int* a = malloc(n*sizeof(int));

	for(int i = 0; i < n; i++){
		a[i] = rand();
	}
	if(a==NULL){
		fprintf(stderr,"Not enough memory");
		exit(EXIT_FAILURE);
	}


	double t = omp_get_wtime();
#pragma omp parallel num_threads(8)
	count_sort(a,n);

	t = 1000*(omp_get_wtime()-t);

	count_sort(a,n);
	printf("time: %.1f ms.\n",t);
	for(int i = 0; i<n;i++){
		printf("%d,", a[i]);
	}
	free(a);
 return 0;
}

/*
 * serial 366 ms
 *
 * parallel 86 ms
 */


