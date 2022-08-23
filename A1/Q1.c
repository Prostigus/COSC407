/*
 ============================================================================
 Name        : Q1.c
 Author      : Prajet Didden
 Version     :
 Copyright   : Your copyright notice
 Description : A1 Q1 in C
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>

int main(void) {
	int a,b,c,d;
	printf("Enter 4 integers separated by spaces:");
	fflush(stdout);
	scanf("%d %d %d %d",&a,&b,&c,&d);
	float average = (a+b+c+d)/4.0f;
	int e[] = {a,b,c,d};
	int count = 0;
	for(int i = 0; i<4;i++){
		if (e[i] > average){
			count++;
		}
	}
	if(count == 0){
		printf("There are no entries above the average(%.1f)",average);
	}
	else if(count ==1){
		printf("There is 1 entry above the average(%.1f)",average);
	}
	else{
		printf("There are %d entry above the average(%.1f)",count,average);
	}
	return EXIT_SUCCESS;
}
