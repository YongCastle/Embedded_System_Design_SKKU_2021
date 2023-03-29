#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>


#define LENGTH 100000000 //134217728 // 2^27



void fill_array(float* vector_arr, int len)
{
	for (int i = 0; i < len; i++)
	{
		vector_arr[i]= 1;
	}
	
	vector_arr[99999999] = 999;
	
	//vector_arr[4095]=100;
}

__global__
void find_max_element(float * A, int numOps)
{
    extern __shared__ float sMem[];
    //load shared MEM
    int tid = threadIdx.x;
	int bid = blockIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index :  0 ~ 1억
	int offset = gridDim.x * blockDim.x;			   // block x thread

	for( int i = 0 ; i < numOps; i ++)
	{
		if(index + (offset) * i < LENGTH)
		{
			sMem[tid] = A[index + (offset) * i];
			__syncthreads();
			for(int s=blockDim.x/2; s>0; s/=2)
			{
				if (tid < s)
				{
					if(sMem[tid] < sMem[tid + s])
					{
						sMem[tid] = sMem[tid + s];
					}
				}
				__syncthreads();
			}
			if(tid == 0)
			{
				A[bid + gridDim.x * i] = sMem[tid];
			}
			__syncthreads();
 		}					
	}
}


int main(int argc, char *argv[])
{
    //input
	int testNum = argc > 1 ? atoi(argv[1]) : 10;
	int block = argc > 2 ? atoi(argv[2]) : 2;
    int thread = argc > 3 ? atoi(argv[3]) : 1024;
	int key = argc > 4? atoi(argv[4]) : 0 ;

	//testNum = 10;

    int numOps;

    //For estimate execution time in gpu
	cudaEvent_t start, stop;
	float time_ms = 0;
	float elapsedtime_sum = 0;


    //Memory declare in Device
    float *dvector_arr = NULL;



    numOps = (LENGTH > (block * thread)) ? ((LENGTH / (block * thread)) + ((LENGTH % (block * thread))? 1 : 0)) : 1;

	printf("------------------------------------------------------------------------------------\n");
	printf("GPU Execution Time Estimatation START\n");
	printf("Input  vector lenghth : %d \n", LENGTH);
    printf("Number of Thread : %d \n", thread);
    printf("Number of Block : %d \n", block);
    printf("Number of Operation : %d \n", (LENGTH / (block * thread)));
	printf("Number of Test : %d \n", testNum);



    
	for (int i = 0; i < testNum; i++)
	{

		int numElements = LENGTH;
		//Memory allocation in HOSt
   		float *hvector_arr = NULL;
    	hvector_arr = (float *) malloc(sizeof(float) * LENGTH);

    	//fill host vector
   		fill_array(hvector_arr, LENGTH);


    	cudaEventCreate(&start);
     	cudaEventCreate(&stop);
      
        cudaMalloc((void **)&dvector_arr, sizeof(float) * LENGTH);

		//Memory transfor Host To Device
		cudaMemcpy(dvector_arr, hvector_arr, sizeof(float) * LENGTH, cudaMemcpyHostToDevice);
		
		//ESTIMATE START
		cudaEventRecord(start, 0);

		while(numElements != 1)
		{
			numOps = (numElements % (block * thread) == 0)? numElements / (block * thread) : numElements / (block * thread) + 1 ;
			//Find MAX Element
			find_max_element<<<block,thread, sizeof(float) * thread>>>(dvector_arr, numOps);

			numElements = (numElements % thread == 0)? numElements / thread : numElements / thread + 1;
		}
			

		//ESTIMATE STOP
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		/*----------------------------- GPU Time Estimate End --------------------------------*/


		cudaMemcpy(hvector_arr, dvector_arr, sizeof(float) * LENGTH, cudaMemcpyDeviceToHost);


		cudaEventElapsedTime( &time_ms, start, stop);
			
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		cudaDeviceSynchronize();


		printf("------------------------------------------------------------------------------------\n");
		printf("TestNum : %2d , elapsedtime = %lf\n", i + 1, time_ms);
		elapsedtime_sum += time_ms;


		cudaFree(dvector_arr);

		printf("The Largest elements in a vector : %f  \n", hvector_arr[0]);
		printf("------------------------------------------------------------------------------------\n");

		if(key == 1)
		{
			for(int i = 0 ; i < LENGTH ; i++)
			{
				if(hvector_arr[i] == 999)
				{
					printf("index = %d\n", i);
				}
			}
			printf("\n\n");
		}
		free(hvector_arr);
	
	}
	printf("\n------------------------------------------------------\n");
	printf("\n Average ElapsedTime = %lf ms \n",elapsedtime_sum / testNum);
	printf("\n----------------------------------------------------------------------\n");


	return 0;

}


