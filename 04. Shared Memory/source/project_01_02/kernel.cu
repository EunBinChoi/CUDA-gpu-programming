#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#define SHARED 1024

__global__ void reduce(int*a ,int *o, int n)
{
	__shared__ int sa[SHARED]; // 공유메모리
	
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x+ threadIdx.x;
	/*
	if(idx > n){
		if(threadIdx.x < SHARED)
		{
			sa[tid] = 0;
			return;
		}
	}*/
	if(idx + (int)n/2 >= n) sa[tid] = a[idx];
	if((n/2) % blockDim.x  == 0) sa[tid] = a[idx] + a[idx + (int)n/2];
	// 나누어떨어질때만 가능
	
	__syncthreads();
	
	// ok
	for(int s = blockDim.x/2; s >= 1; s >>= 1){
		if(tid < s)
			sa[tid] = sa[tid] + sa[tid + s];
		__syncthreads();
	}
	
	if(tid == 0) o[blockIdx.x] = sa[tid];
}

int main(){
	const int N = 1000*1024;
	int block_size = 1024;
	int block_num = (N + block_size -1)/block_size;
	
	int *a, *o;
	a = (int *)malloc(sizeof(int) * N);
	o = (int *)malloc(sizeof(int) * block_num);
	for(int i = 0 ; i < N; i++) {
		a[i] = (i % 2 ==0) ? i : -i;
		//printf("%d ", a[i]);
	}

	int *id, *od;
	cudaMalloc((void **)&id, sizeof(int)*N);
	cudaMalloc((void **)&od, sizeof(int)*block_num);
	
	cudaEvent_t start, end;
	float gpu_time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMemcpy(id,a,sizeof(int)*N, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	reduce<<<block_num/2, block_size>>>(id,od,N);
	
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end);
	
	printf("\n>>gpu_time = %f\n", gpu_time);
	//printf("%d\n\n", N/2);
	cudaMemcpy(o,od,sizeof(int)*block_num, cudaMemcpyDeviceToHost);
	int sum = 0;
	for(int i = 0 ; i < block_num; i++){
		//printf("%d ", o[i]);
		sum += o[i];
	}
	printf("\n");
	printf("GPU : %d", sum);
	printf("\n");

	//CPU
	int sum2 = 0;
	for(int i = 0 ; i < N ; i ++)
	{
		sum2 += a[i];
	}
	
	printf("CPU : %d\n",sum2);

	cudaError_t err;
	err = cudaGetLastError();
	printf("%s\n", cudaGetErrorString(err));

	free(a); free(o);
	cudaFree(id); cudaFree(od);

	return 0;
}