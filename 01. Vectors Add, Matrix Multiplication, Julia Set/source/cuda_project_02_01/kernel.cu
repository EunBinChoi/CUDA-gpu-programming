
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

//#define NUM_DATA 123456
using namespace cv;


__global__ void mat_mul(float *a, float *b, float *c, int M, int N, int K){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;

	for(int k = 0; k < K; k++){
		sum += a[row*K+k] * b[k*N+col];
	}
	c[row*N + col] = sum;
}

int main(){
	// 실습 2-1
	cudaEvent_t start, end;
	float gpu_time; 
	
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	const int M = 2048;
	const int N = 1536;
	const int K = 1024;

	float *a, *b, *c;
	a = (float *)malloc(sizeof(float) * M*K);
	b = (float *)malloc(sizeof(float) * K*N);
	c = (float *)malloc(sizeof(float) * M*N);

	for(int i = 0; i < M*K ; i++)
		a[i] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2); 
	for(int i = 0; i < K*N ; i++)
		b[i] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	  
	float *da, *db, *dc;
	cudaMalloc((void**)&da, sizeof(float) * M*K);
	cudaMalloc((void**)&db, sizeof(float) * K*N);
	cudaMalloc((void**)&dc, sizeof(float) * M*N);
	
	cudaMemcpy(da, a, sizeof(float)*M*K,cudaMemcpyHostToDevice);
	cudaMemcpy(da, b, sizeof(float)*K*N,cudaMemcpyHostToDevice);
	
	int thread = 0;
	printf("block 당 thread 개수 입력 : ");
	scanf("%d",&thread);
	
	cudaEventRecord(start, 0);
	dim3 dim_block(thread, thread, 1);
	dim3 dim_grid(N/thread, M/thread, 1);
	mat_mul<<<dim_grid, dim_block>>>(da,db,dc,M,N,K);

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end); // 시간 간격을 잼

	cudaMemcpy(c, dc, sizeof(float)*M*N,cudaMemcpyDeviceToHost);
	printf("gpu time = %f\n", gpu_time);
	
	free(a); free(b); free(c);
	cudaFree(da); cudaFree(db), cudaFree(dc);

	return 0;
}


