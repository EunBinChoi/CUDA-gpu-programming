
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

__global__ void mat_mul(float *a, float *b, float *c, int M, int N, int K){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;

	for(int k = 0; k < K; k++){
		sum += a[row*K+k] * b[k*N+col];
	}

	c[row*N + col] = sum;
}

void nmat_mul(float *a, float *b, float *c, int M, int N, int K){

	float sum = 0;

	for(int i = 0 ; i < M ; i++) {
		for(int j = 0 ; j < N ; j++) {
			for(int k = 0; k < K; k++) {
				sum += a[i*K+k] * b[k*N+j];
			}
			c[i*N+j] = sum;
			sum = 0;
		}
	}
}

int main(){

	// 실습 2-3(쓰레드의 개수가 2,4일 때 작동되지 않음)
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

	int thread;
	//2,4,6,8,12,16,24,32

	//i가 2,4일때 작동되지 않음
	for(int i = 4 ; i <= 32 ; ){
		Sleep(1000);
		gpu_time = 0;
		thread = i;
		
		cudaEventRecord(start, 0);
		// 2차원 배열 => dim3
		dim3 dim_blocks(thread, thread, 1);
		dim3 dim_grid(N/thread, M/thread, 1);
		mat_mul<<<dim_grid, dim_blocks>>>(da,db,dc,M,N,K);
		
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&gpu_time, start, end);
		//printf("%f\n", gpu_time);
		cudaMemcpy(c, dc, sizeof(float)*M*N,cudaMemcpyDeviceToHost);

		printf("thread = %d, gpu time = %.10f\n",thread, gpu_time);

		if(i < 8) i = i + 2;
		else if(i < 16) i = i + 4;
		else if(i < 32) i = i + 8;
		else break;
	}

	free(a); free(b); free(c);
	cudaFree(da); cudaFree(db); cudaFree(dc);

	return 0;

}

