
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

//#define NUM_DATA 123456
__global__ void mat_mul(float *a, float *b, float *c, int M, int N, int K){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;

	for(int k = 0; k < K; k++){
		sum += a[row*K+k] * b[k*N+col];
	}

	c[row*N + col] = sum;
}

// CPU에서 행렬의 곱셈
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

		//// 실습 2-2
		const int M = 2048;
		const int N = 1536;
		const int K = 1024;
	
		/*int M = 10;
		int N = 10;
		int K = 10;
	*/
		float *a, *b, *c, *c_cpu;
		a = (float *)malloc(sizeof(float) * M*K);
		b = (float *)malloc(sizeof(float) * K*N);
		c = (float *)malloc(sizeof(float) * M*N);
		c_cpu = (float *)malloc(sizeof(float) * M*N);
	
		//for(int i = 0 ; i < M*K ;i++){a[i]=2;}
		//for(int i = 0 ; i < K*N ;i++){b[i]=3;}
	
		for(int i = 0; i < M*K ; i++)
			a[i] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2); 
		for(int i = 0; i < K*N ; i++)
			b[i] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);
	
		float *da, *db, *dc;
		cudaMalloc((void**)&da, sizeof(float) * M*K);
		cudaMalloc((void**)&db, sizeof(float) * K*N);
		cudaMalloc((void**)&dc, sizeof(float) * M*N);
	
		cudaMemcpy(da, a, sizeof(float)*M*K,cudaMemcpyHostToDevice);
		cudaMemcpy(db, b, sizeof(float)*K*N,cudaMemcpyHostToDevice);
	
		int thread = 0;
		printf("block 당 thread 개수 입력 : ");
		scanf("%d",&thread);
	
		dim3 dim_block(thread, thread);
		dim3 dim_grid(N/thread, M/thread);
		mat_mul<<<dim_grid, dim_block>>>(da,db,dc,M,N,K);
		cudaMemcpy(c, dc, sizeof(float)*M*N,cudaMemcpyDeviceToHost);		
		
		nmat_mul(a, b, c_cpu, M, N, K);

		int i = 0;
		for(i = 0 ; i < M*N ; i++){
			if(fabsf(c_cpu[i]-c[i]) > 0.0001){ 
				printf("=>[ false ]\n");
				break;
			}
		}
		if(i == M*N) printf("=>[ 모두 일치합니다 ]\n");
	
		free(a); free(b); free(c); free(c_cpu);
		cudaFree(da); cudaFree(db); cudaFree(dc);
		
		return 0;
}

	