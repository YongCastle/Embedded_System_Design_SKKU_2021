#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define LENGTH 100000000


struct Element_info {
	float value;
	int index;
};


void fill_array(float* vector_arr, int len)
{
	for (int i = 0; i < len; i++)
	{
		vector_arr[i] = 10;
	}

	for (int i = 0; i < len; i++)
	{
		if (vector_arr[i] != 10)
		{
			printf("error");
		}
	}

	vector_arr[43254] = 100;
	
}


void find_max(struct Element_info * max_element, const float* vector_arr, int len)
{
	float max_value = vector_arr[0];
	int max_index = 0;
	

	for (int i = 0; i < len; i++)
	{
		if (max_value < vector_arr[i])
		{
			max_value = vector_arr[i];
			max_index = i;
		}
	}

	max_element->value = max_value;
	max_element->index = max_index;

}


int main(int argc, char *argv[])
{
	float *vector_arr;
	struct Element_info *max_element;
	int testNum = argc > 1 ? atoi(argv[1]) : 10;

	struct timeval startTime, endTime;
	double elapsedTime;
	double elapsedTime_sum = 0;

	max_element = (struct Element_info *) malloc(sizeof(struct Element_info));


	printf("------------------------------------------------------------------------------------\n");
	printf("CPU Execution Time Estimatation START\n");
	printf("Input  vector lenghth : %d \n", LENGTH);
	printf("Number of Test : %d \n", testNum);
	
	
	for (int i = 0; i < testNum; i++)
	{

		vector_arr = (float *) malloc(sizeof(float) * LENGTH);
		//fill vector
		fill_array(vector_arr, LENGTH);

		/*--------------------------------------------- CPU TIME ESTIMATE START -----------------------------*/
		gettimeofday(&startTime, NULL);

		find_max(max_element, vector_arr, LENGTH);

		gettimeofday(&endTime, NULL);
		/*--------------------------------------------- CPU TIME ESTIMATE END -------------------------------*/

		elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000.00 + (endTime.tv_usec - startTime.tv_usec) / 1000.00;
		elapsedTime_sum += elapsedTime;
		printf(" Test : %d , Elapsed Time :  %lf ms\n", i + 1, elapsedTime);

		free(vector_arr);
	}
	printf("------------------------------------------------------------------------------------\n");
	printf("\n Average Elapsed Time : %lf ms \n", elapsedTime_sum / testNum);
	printf("------------------------------------------------------------------------------------\n");



	
	printf("The Largest elements in a vector : %f  and index : %d \n", max_element->value, max_element->index);
	printf("------------------------------------------------------------------------------------\n\n");



	return 0;
}


