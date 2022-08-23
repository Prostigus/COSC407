#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define NRA 200
#define NCA 300
#define NCB 100

/**
 * Hello OpenMP World prints the number of threads and the current thread id
 */

int main (int argc, char *argv[]) {

int i,j,k;
double a[NRA][NCA], b[NCA][NCB], c[NRA][NCB];
int n = 16;
double t = omp_get_wtime();
#pragma omp parallel num_threads(n)
{

#pragma omp for collapse(2)
for(i=0;i<NRA;i++){
	for(j=0;j<NCA;j++)
		a[i][j] = i+j;
}
#pragma omp for collapse(2)
for(i=0;i<NCA;i++){
	for(j=0;j<NCB;j++)
		b[i][j] = i*j+1;
}


//multiplication'
#pragma omp for collapse(3)
for(i=0;i<NRA;i++){
	for(j=0;j<NCB;j++){
		for(k=0;k<NCA;k++){
			c[i][j]+=a[i][k] * b[k][j];
		}
	}
}
}

t = 1000 * (omp_get_wtime() - t);
printf("Finished in %.1f ms.\n", t);
/*
for (int i = 0; i < NRA; i++) {
		for (int j = 0; j < NCB; j++){
				printf("%6.2f ", c[i][j]);
		}
		printf("\n");
	}
*/

 return 0;
}

/*
 * parallel code runs much faster with more threads as the problem size grows
 * runs slower with many threads when the problem size is relativly small
 */
