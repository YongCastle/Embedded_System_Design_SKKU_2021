#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define MAX_SHAREDSIZEperBLOCK 12288
#define BLOCK_SIZE 512


__constant__ int cdata[5];

__host__ void matrix_fill(int* matrix, int x, int y);
__host__ void matrix_printf(const int* matrix, int x, int y);
__host__ int matrix_check(const int* matrix, int x, int y);



__global__
void device_matrix_concat(int * C, int * A, int * B)
{

	
	/*
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	//int index2 = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	int tid = threadIdx.x;

	extern __shared__ int sMem[];
	
	//printf("%d %d %d %d %d", cdata[0], cdata[1], cdata[2], cdata[3], cdata[4]);
	//cdata[] = {numOps, M, P , Q, sizeA};
	
	for (int i = 0 ; i < cdata[0]; i++)
	{
		if(index < cdata[4]) 
		{	
			sMem[tid] = A[index];
		}
		else
		{
			sMem[tid] = B[index - cdata[4]];

		}
		__syncthreads();

		if(index < cdata[4]) 
		{	
			C[(index / cdata[2]) * (cdata[2] + cdata[3]) + (index % cdata[2])] = sMem[tid];
		}
		else
		{
			C[((index - cdata[4]) / cdata[3]) * (cdata[2] + cdata[3]) + (cdata[2]+ (index - cdata[4]) % cdata[3])] = sMem[tid];
		}
		index += offset;
		
	}
	*/



	/*
	for (int i = 0 ; i < cdata[0]; i++)
	{
		if(index2 < cdata[4]) 
		{	
			C[(index2 / cdata[2]) * (cdata[2] + cdata[3]) + (index2 % cdata[2])] = sMem[tid];
		}
		else
		{
			C[((index2 - cdata[4]) / cdata[3]) * (cdata[2] + cdata[3]) + (cdata[2]+ (index2 - cdata[4]) % cdata[3])] = sMem[tid];
		}
		index2 += offset;
	}
	__syncthreads();
	*/
	
	
	int index2 = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;

	for (int i = 0 ; i < cdata[0]; i++)
	{
		if(index2 < cdata[4]) 
		{	
			C[(index2 / cdata[2]) * (cdata[2] + cdata[3]) + (index2 % cdata[2])] = A[index2];
		}
		else
		{
			C[((index2 - cdata[4]) / cdata[3]) * (cdata[2] + cdata[3]) + (cdata[2]+ (index2 - cdata[4]) % cdata[3])] = B[index2-cdata[4]];
		}
		index2 += offset;
	}
	
	

	/*
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int offset = gridDim.x * blockDim.x;
	

	for (int i = 0 ; i < cdata[0]; i++)
	{
		if( index < cdata[1] * (cdata[2] + cdata[3]))
		{
			if(index % (cdata[2] +cdata[3]) < cdata[2])
			{
				C[index] = A[cdata[2] * (index / (cdata[2] + cdata[3])) + index % (cdata[2] +cdata[3])];
				//printf("HEllo  index : %d, A[%d] :)\n", index, p * (index / (p + q)) + index % (p + q) );
			}
			else
			{
				C[index] = B[cdata[3] * (index / (cdata[2]+ cdata[3])) + index % (cdata[2]+ cdata[3]) - cdata[2]];
				//printf("HEllo  index : %d, B[%d] :)\n", index, q * (index / (p + q)) + index % (p + q) - p);
			}	
			index += offset;
		}
	}
	*/

}

int main(int argc, char *argv[])
{

	
	//host_matrix
	int* host_matrixA = NULL;
	int* host_matrixB = NULL;
	int* host_matrixC = NULL;

	

	//device_matrix
	int* device_matrixA = NULL;
	int* device_matrixB = NULL;
	int* device_matrixC = NULL;



	//GPU TIME
	cudaEvent_t start, stop;
	float time_ms = 0;
	float elapsedtime_sum = 0;
	float optimaltime = 0;

	int block = atoi(argv[1]);
	int thread = atoi(argv[2]);
	int testNum = argc > 3 ? atoi(argv[3]) : 1;
	int M = argc > 4 ? atoi(argv[4]) : 512;
	int P = argc > 5 ? atoi(argv[5]) : 1024;
	int Q = argc > 6 ? atoi(argv[6]) : 512;
	int key = 0; 								// matrix print?
	key = argc > 7 ? atoi(argv[7]) : 0;
	int findOptimal = 0;
	findOptimal = argc > 8 ? atoi(argv[8]) : 0;
	int numElements = M * (P + Q);

	int sizeA = M * P;
	int sizeB = M * Q;
	int sizeC = M * (P + Q);
	int numOps;
	int optimalblock;
	int optimalthread;
	int t;
	int b;

	

	//Memory Allocatation
	host_matrixA = (int *)malloc(sizeof(int) * sizeA);
	host_matrixB = (int *)malloc(sizeof(int) * sizeB);
	host_matrixC = (int *)malloc(sizeof(int) * sizeC);

	//host_fill matrix
	matrix_fill(host_matrixA, M, P);
	matrix_fill(host_matrixB, M, Q);


	if(findOptimal == 0)
	{
		numOps = (numElements > (block * thread)) ? ((numElements / (block * thread)) + ((numElements % (block * thread))? 1 : 0)) : 1;
		
		printf("\n----------------------------------------------------------------------\n");
		printf("GPU Execution Time Estimatation START\n");
		printf("Input  Matrix : %4d x %-4d \n", M, P);
		printf("                %4d x %-4d \n", M, Q);
		printf("Output Matrix : %4d x %-4d \n", M, (P + Q));
		printf("\n");
		printf("Number of Blocks    :  %4d\n", block);
		printf("Number of Threads   :  %4d\n", thread);
		printf("Number of Operation :  %4d\n", numOps);
		printf("Number of testnum   :  %4d\n", testNum);
		printf("----------------------------------------------------------------------\n");

		for(int i = 0; i < testNum ; i++)
  		{
   
    		cudaEventCreate(&start);
     		cudaEventCreate(&stop);
      
			cudaMalloc((void **)&device_matrixA, sizeof(int) * sizeA);
			cudaMalloc((void **)&device_matrixB, sizeof(int) * sizeB);
			cudaMalloc((void **)&device_matrixC, sizeof(int) * sizeC);

			int data[] = {numOps, M, P , Q, sizeA};
			

			//Add for HW2, constant memory value transfer to device
			cudaMemcpyToSymbol(cdata, &data, sizeof(int) * 5);
		

			//Memory transfor Host To Device
			cudaMemcpy(device_matrixA, host_matrixA, sizeof(int) * sizeA, cudaMemcpyHostToDevice);
			cudaMemcpy(device_matrixB, host_matrixB, sizeof(int) * sizeB, cudaMemcpyHostToDevice);

			//ESTIMATE START
			cudaEventRecord(start, 0);

			//concat in GPU
			device_matrix_concat<<<block,thread>>>(device_matrixC, device_matrixA, device_matrixB);

			//ESTIMATE STOP
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			/*----------------------------- GPU Time Estimate End --------------------------------*/

			cudaMemcpy(host_matrixC, device_matrixC, sizeof(int) * sizeC, cudaMemcpyDeviceToHost);


			cudaEventElapsedTime( &time_ms, start, stop);
			
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			cudaDeviceSynchronize();
			//Memory transfor Device To Host


			printf("TestNum : %2d , elapsedtime = %lf\n", i + 1, time_ms);
			elapsedtime_sum += time_ms;


			cudaFree(device_matrixA);
			cudaFree(device_matrixB);
			cudaFree(device_matrixC);

		}
		printf("\n------------------------------------------------------\n");
		printf("\n Average ElapsedTime = %lf ms \n",elapsedtime_sum / testNum);
		printf("\n----------------------------------------------------------------------\n");

	}


	if(findOptimal == 1)
	{
		printf("\n----------------------------------------------------------------------\n");
		printf("FIND OPTIMAL BLOCKS & THREADS START\n");
		printf("Input  Matrix : %4d x %-4d \n", M, P);
		printf("                %4d x %-4d \n", M, Q);
		printf("Output Matrix : %4d x %-4d \n", M, (P + Q));
		printf("\n\n");
		printf("Wait a Minuite, This process needs a lot of time \n\n");
		for(int e = 1; e <= 32 ; e ++)	// thread loop
		{
			t = e * 32;		//thread = 32 , 64 , ... ,1024
			b = ((512 * (1024 + 512)) % t == 0)? 512 * (1024 + 512) / t : 512 * (1024 + 512) / t + 1;

			elapsedtime_sum = 0;

			for(int i = 0; i < testNum ; i++)
			{
					
				numOps = (numElements > (b * t)) ? ((numElements / (b * t)) + ((numElements % (b * t))? 1 : 0)) : 1;

				cudaEventCreate(&start);
				cudaEventCreate(&stop);
					
				cudaMalloc((void **)&device_matrixA, sizeof(int) * sizeA);
				cudaMalloc((void **)&device_matrixB, sizeof(int) * sizeB);
				cudaMalloc((void **)&device_matrixC, sizeof(int) * sizeC);

				//Add for HW2, constant memory value transfer to device
				int data[] = {numOps, M, P , Q, sizeA};
				

				//Add for HW2, constant memory value transfer to device
				cudaMemcpyToSymbol(cdata, &data, sizeof(int) * 5);
					

				//Memory transfor Host To Device
				cudaMemcpy(device_matrixA, host_matrixA, sizeof(int) * sizeA, cudaMemcpyHostToDevice);
				cudaMemcpy(device_matrixB, host_matrixB, sizeof(int) * sizeB, cudaMemcpyHostToDevice);

				//ESTIMATE START
				cudaEventRecord(start, 0);

				//concat in GPU
				device_matrix_concat<<<block,thread>>>(device_matrixC, device_matrixA, device_matrixB);
				
				//ESTIMATE STOP
				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaDeviceSynchronize();


				/*----------------------------- GPU Time Estimate End --------------------------------*/

				cudaMemcpy(host_matrixC, device_matrixC, sizeof(int) * sizeC, cudaMemcpyDeviceToHost);
					


				cudaEventElapsedTime( &time_ms, start, stop);
					
				cudaEventDestroy(start);
				cudaEventDestroy(stop);
					
				//Memory transfor Device To Host


				//printf("TestNum : %2d , elapsedtime = %lf\n", i + 1, time_ms);
				elapsedtime_sum += time_ms;


				cudaFree(device_matrixA);
				cudaFree(device_matrixB);
				cudaFree(device_matrixC);

			}

			//printf("\n------------------------------------------------------\n");
			//printf("\n Average ElapsedTime = %lf ms \n",elapsedtime_sum / testNum);
			//printf("\n----------------------------------------------------------------------\n");


			if (t == 32)
			{
				optimaltime = (elapsedtime_sum / testNum);
			}
			if ((elapsedtime_sum / testNum) < optimaltime)
			{
				optimaltime = (elapsedtime_sum / testNum);
				optimalblock = b;
				optimalthread = t;
			}

			printf(" Block : %5d , Thread : %5d ==> Average Elapsed Time = %7lf ms \n", b, t, (elapsedtime_sum / testNum));

		}	
		
		numOps = (numElements > (optimalblock * optimalthread)) ? ((numElements / (optimalblock * optimalthread)) + ((numElements % (optimalblock * optimalthread))? 1 : 0)) : 1;
		printf("Number of testnum   :  %4d\n", testNum);
		printf("Number of OptimalBlocks    :  %4d\n", optimalblock);
		printf("Number of OptimalThreads   :  %4d\n", optimalthread);
		printf("Number of Operation :  %4d\n", numOps);
		printf("----------------------------------------------------------------------\n");
		printf("\n------------------------------------------------------\n");
		printf("\n Optimal Average ElapsedTime = %lf ms \n",optimaltime);
		printf("\n----------------------------------------------------------------------\n");

	}



	//print matrix
	if (key != 0) 
	{
		printf("\n      <Host_Matrix A>\n");
		matrix_printf(host_matrixA, M, P);

		printf("\n      <Host_Matrix B>\n");
		matrix_printf(host_matrixB, M, Q);

		printf("\n      <Host_Matrix C>\n");
		matrix_printf(host_matrixC, M, P + Q);
	}

	free(host_matrixA);
	free(host_matrixB);
	free(host_matrixC);


	return 0;
}

__host__
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

__host__
void matrix_printf(const int* matrix, int x, int y)
{
    for (int i = 0; i < x; i++)
    {
        for (int j = 0; j < y; j++)
        {

			printf("%6d ", matrix[y * i + j]);
			
        }
        printf("\n");
    }
}

