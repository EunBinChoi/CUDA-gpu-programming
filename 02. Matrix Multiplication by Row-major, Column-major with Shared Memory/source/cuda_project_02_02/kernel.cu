// 공유 메모리를 이용한 행렬곱셈 커널을 작성하시오
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

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
/*
__global__ void shared_mat_mul(float *a, float *b, float *c, int M, int N, int K){
__shared__ float tile_a[32][32];
__shared__ float tile_b[32][32];

int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int xid = threadIdx.x;
int yid = threadIdx.y;

float sum = 0;

for(int k = 0; k < K/32; k++){
tile_a[yid][xid] = a[row*K + (k*32+xid)];
tile_b[yid][xid] = b[(k*32+yid)*N+col];

__syncthreads();

for(int i = 0; i < 32;i++){
sum += tile_a[yid][i] * tile_b[i][xid];
}
__syncthreads();
}

c[row*N + col] = sum;
}
*/

__global__ void shared_mat_mul(float *a, float *b, float *c, int M, int N, int K) {
	__shared__ float tile_a[32][32];
	__shared__ float tile_b[32][32];
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	float sum = 0;
	//if(row >= M) return;
	//if(col >= N) return;

	for(int i=0;i<(K+31)/32;i++) {
		if((row >= M) || ((i*32+threadIdx.x) >= K))
			tile_a[threadIdx.y][threadIdx.x] = 0;
		else
			tile_a[threadIdx.y][threadIdx.x] = a[row*K + (i*32+threadIdx.x)];
		if((col >= N) || ((i*32+threadIdx.y) >= K))
			tile_b[threadIdx.y][threadIdx.x] = 0;
		else
			tile_b[threadIdx.y][threadIdx.x] = b[(i*32+threadIdx.y)*N + col];
		__syncthreads();

		if((row < M) && (col < N))
			for(int k=0;k<32;k++)
				sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
		__syncthreads();
	}
}
int main(){
	// 실습 2-1
	//cudaEvent_t start, end;
	//float gpu_time; 
	//
	//cudaEventCreate(&start);
	//cudaEventCreate(&end);

	const int M = 2048;
	const int N = 1536;
	const int K = 1024;

	float *a, *b, *c, *shared_c;
	a = (float *)malloc(sizeof(float) * M*K);
	b = (float *)malloc(sizeof(float) * K*N);
	c = (float *)malloc(sizeof(float) * M*N);
	shared_c = (float *)malloc(sizeof(float) * M*N);

	for(int i = 0; i < M*K ; i++)
		a[i] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2); 
	for(int i = 0; i < K*N ; i++)
		b[i] = (float)(rand() - RAND_MAX/2)/(RAND_MAX/2);

	float *da, *db, *dc, *shared_dc;
	cudaMalloc((void**)&da, sizeof(float) * M*K);
	cudaMalloc((void**)&db, sizeof(float) * K*N);
	cudaMalloc((void**)&dc, sizeof(float) * M*N);
	cudaMalloc((void**)&shared_dc, sizeof(float) * M*N);

	cudaMemcpy(da, a, sizeof(float)*M*K,cudaMemcpyHostToDevice);
	cudaMemcpy(da, b, sizeof(float)*K*N,cudaMemcpyHostToDevice);

	int thread = 0;
	printf("block 당 thread 개수 입력 : ");
	scanf("%d",&thread);

	//cudaEventRecord(start, 0);
	dim3 dim_block(thread, thread, 1);
	dim3 dim_grid(N/thread, M/thread, 1);
	mat_mul<<<dim_grid, dim_block>>>(da,db,dc,M,N,K);
	shared_mat_mul<<<dim_grid, dim_block>>>(da,db,shared_dc,M,N,K);
	//cudaEventRecord(end, 0);
	//cudaEventSynchronize(end);
	//cudaEventElapsedTime(&gpu_time, start, end); // 시간 간격을 잼

	cudaMemcpy(c, dc, sizeof(float)*M*N,cudaMemcpyDeviceToHost);
	cudaMemcpy(shared_c, shared_dc, sizeof(float)*M*N,cudaMemcpyDeviceToHost);

	//printf("gpu time = %f\n", gpu_time);
	int i = 0;
	for(i = 0 ; i < M*N ; i++){
		if(fabsf(c[i]-shared_c[i]) > 0.0001){ 
			printf("=>[ false ]\n");
			break;
		}

	}
	if(i == M*N) printf("=>[ 모두 일치합니다 ]\n");


	free(a); free(b); free(c);
	cudaFree(da); cudaFree(db), cudaFree(dc);

	return 0;
}


