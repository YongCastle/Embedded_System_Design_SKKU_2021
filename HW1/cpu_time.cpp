#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>



void matrix_fill(int* matrix, int x, int y);
void matrix_concat(int* matrix3, const int* matrix1, const int* matrix2, int m, int p, int q);
void matrix_printf(const int* matrix, int x, int y);

int main(int argc, char *argv[])
{
	int* matrix1 = NULL;
	int* matrix2 = NULL;
	int* matrix3 = NULL;

	int M = argc > 1 ? atoi(argv[1]) : 512;
	int P = argc > 2 ? atoi(argv[2]) : 1024;
	int Q = argc > 3 ? atoi(argv[3]) : 512; 
	int testNum = argc > 4 ? atoi(argv[4]) : 10;
	int printkey = argc > 5 ? atoi(argv[5]) : 0;

	struct timeval startTime, endTime;
	double elapsedTime;
	double elapsedTime_sum = 0;

	matrix1 = (int *)malloc(sizeof(int) * M * P);
	matrix2 = (int *)malloc(sizeof(int) * M * Q);
	matrix3 = (int *)malloc(sizeof(int) * M * (P + Q));


	//fill matrix
	matrix_fill(matrix1, M, P);
	matrix_fill(matrix2, M, Q);



  	printf("------------------------------------------------------------------------------------\n");
	printf("CPU Execution Time Estimatation START\n");
	printf("Input  Matrix : %d x %d \n", M, P);
	printf("                %d x %d \n", M, Q);
	printf("Output Matrix : %d x %d \n", M, (P + Q));
  	printf("Number of Test : %d \n", testNum);


	for (int i = 0 ; i < testNum ; i++)
	{

  		/*--------------------------------------------- CPU TIME ESTIMATE START -----------------------------*/
	  	gettimeofday(&startTime, NULL);

	  	//concat
  		matrix_concat(matrix3, matrix1, matrix2, M, P, Q);

	  	gettimeofday(&endTime, NULL);
  		/*--------------------------------------------- CPU TIME ESTIMATE END -------------------------------*/

	  	elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000.00 + (endTime.tv_usec - startTime.tv_usec) / 1000.00 ;
		elapsedTime_sum += elapsedTime;
  		printf(" Test : %d , Elapsed Time :  %lf ms\n", i + 1,  elapsedTime);
	}
  	printf("------------------------------------------------------------------------------------\n");
  	printf("\n Average Elapsed Time : %lf ms \n" , elapsedTime_sum / testNum); 
  	printf("------------------------------------------------------------------------------------\n\n");


	if( printkey != 0)
  	{
		//print matrix
		printf("\nMatrix A\n");
		matrix_printf(matrix1, M, P);
	   	printf("\nMatrix B\n");
		matrix_printf(matrix2, M, Q);
	    	printf("\nMatrix C\n");
		 matrix_printf(matrix3, M, P + Q);
	}

	free(matrix1);
	free(matrix2);
	free(matrix3);

  	printf("------------------------------------------------------------------------------------\n");

	return 0;
}
void matrix_fill(int* matrix, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0 ; j < y; j ++)
        {
            matrix[y * i + j] = y * i + j;
        }
    }
}

void matrix_concat(int* matrix3, const int* matrix1, const int* matrix2, int m, int p, int q)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            matrix3[(p + q) * i + j] = matrix1[p * i + j];
        }
        for (int k =0; k < q; k++)
        {
            matrix3[(p + q) * i + (p + k)] = matrix2[q * i + k];
        }
    }
}
void matrix_printf(const int* matrix, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {
            printf("%3d  ", matrix[y * i + j]);
        }
        printf("\n");
    }
}
