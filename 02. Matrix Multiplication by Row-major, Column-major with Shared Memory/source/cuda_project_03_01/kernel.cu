#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

__global__ void gpu_transpose(int *a, int* b, int M, int N){	

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	b[col*M + row] = a[row*N + col];
}

int main(){
	cudaEvent_t start, end;
	float gpu_time; 
	
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	const int M = 2048;
	const int N = 1024;
	
	int *a, int* b;
	a = (int *)malloc(sizeof(int) * M*N);
	b = (int *)malloc(sizeof(int) * M*N);

	for(int i =0 ; i < M*N; i++) 
		a[i] = (int)(rand()%100);

	int *da,*db;
	cudaMalloc((void**)&da, sizeof(int) * M*N);
	cudaMalloc((void**)&db, sizeof(int) * M*N);
	
	cudaMemcpy(da, a, sizeof(int)*M*N,cudaMemcpyHostToDevice);
	//cudaMemcpy(db, b, sizeof(float)*M*N,cudaMemcpyHostToDevice);
	
	int thread = 32;
	cudaEventRecord(start, 0);
	dim3 dim_block(thread, thread, 1);
	dim3 dim_grid(N/thread, M/thread, 1);
	gpu_transpose<<<dim_grid, dim_block>>>(da,db,M,N);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end); // 시간 간격을 잼

	
	cudaMemcpy(b, db, sizeof(int)*M*N,cudaMemcpyDeviceToHost);
	
	printf("블럭당 thread 개수 = %d, gpu time = %f\n", thread, gpu_time);
	
	int i = 0, j = 0;

	printf("\n");
	for(i = 0 ; i < M ; i++){
		for(j = 0; j < N ; j++){
			if(abs(a[i*N+j]-b[j*M+i]) > 0.0001){
				printf("=>[ false ]\n");
				return 0;
			}
		}
	}
	if(i*j == M*N) printf("=>[ 모두 일치합니다 ]\n");
	return 0;
}


