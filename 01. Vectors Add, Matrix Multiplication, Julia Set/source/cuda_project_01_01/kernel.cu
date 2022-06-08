#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

//#define NUM_DATA 123456
//using namespace cv;

__global__ void vec_add(float *a, float *b, float *c, int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= n) return;
	c[tid] = a[tid] + b[tid];
}

int main(){
	// �ǽ� 1-1
	int data, thread; // data:�� �������� ��, thread:�������� ����
	cudaEvent_t start,end;
	float gpu_time;
	printf("�� data�� ������ ���� thread ������ �Է��Ͻÿ�(ex. 123456, 1024) : ");
	scanf("%d, %d", &data, &thread);

	cudaEventCreate(&start);
	cudaEventCreate(&end);

	float *a,*b,*c;
	a = (float *)malloc(sizeof(float) * data);
	b = (float *)malloc(sizeof(float) * data);
	c = (float *)malloc(sizeof(float) * data);

	for(int i = 0 ; i < data ; i++){
		a[i] = i*2; // ������ ��
		b[i] = i+3; // ������ ��
	}

	int block_num = data / thread;

	float *da, *db, *dc;
	cudaMalloc((void**)&da, sizeof(float)*data);
	cudaMalloc((void**)&db, sizeof(float)*data);
	cudaMalloc((void**)&dc, sizeof(float)*data);

	cudaMemcpy(da,a,sizeof(float)*data,cudaMemcpyHostToDevice);
	cudaMemcpy(db,b,sizeof(float)*data,cudaMemcpyHostToDevice);
	
	cudaEventRecord(start, 0);
	vec_add<<<block_num,thread>>>(da,db,dc,data);
	// block_num : �� ����
	// thread : ���� ������ ����
	cudaEventRecord(end,0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end); // �� ���� �ð��� �󸶳� �귶����
	
	cudaMemcpy(c, dc, sizeof(float)*data, cudaMemcpyDeviceToHost);

	printf("\n");
	printf("gpu time = %f\n", gpu_time);

	free(a);free(b);free(c);
	cudaFree(da);cudaFree(db);cudaFree(dc);
	
	return 0;
}

