#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

//#define NUM_DATA 123456
using namespace cv;

__global__ void vec_add(float *a, float *b, float *c, int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= n) return;
	c[tid] = a[tid] + b[tid];
}

// 실습 1-2
int main(){
	int data = 123456, thread; // data : 데이터의 개수, thread : 쓰레드의 개수
	cudaEvent_t start,end;
	float gpu_time;

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	float *a, *b, *c;
	a = (float *)malloc(sizeof(float) * data);
	b = (float *)malloc(sizeof(float) * data);
	c = (float *)malloc(sizeof(float) * data);

	for(int i = 0 ; i < data ; i++){
		a[i] = i*2;
		b[i] = i+3;
	}

	float *da, *db, *dc;
	cudaMalloc((void **)&da, sizeof(float)*data);
	cudaMalloc((void **)&db, sizeof(float)*data);
	cudaMalloc((void **)&dc, sizeof(float)*data);

	cudaMemcpy(da, a, sizeof(float)*data,cudaMemcpyHostToDevice);
	cudaMemcpy(da, b, sizeof(float)*data,cudaMemcpyHostToDevice);

	int blocknum;

	for(thread = 2 ; thread <= 1024 ; thread=thread*2){
		gpu_time = 0;
		blocknum = data/thread;

		cudaEventRecord(start, 0);
		vec_add<<<blocknum, thread>>>(da,db,dc,data);
		cudaEventRecord(end, 0);

		cudaEventSynchronize(end);
		cudaEventElapsedTime(&gpu_time, start, end);
		cudaMemcpy(da, c, sizeof(float)*data,cudaMemcpyDeviceToHost);

		printf("블럭당 thread 개수 = %d, gpu time = %f\n", thread, gpu_time);
	}

	free(a);free(b);free(c);
	cudaFree(da);cudaFree(db);cudaFree(dc);
	return 0;
}