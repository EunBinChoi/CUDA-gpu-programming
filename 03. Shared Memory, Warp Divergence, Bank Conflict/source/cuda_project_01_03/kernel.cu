// reduce 2 ���α׷� ����
// reduce2 => ���� �б� �ذ�
// 0 2 4 ... => ���ϴ� thread, 1 3 5 ... => ��� thread
// ���� �бⰡ �Ͼ�� �ʵ��� �Ѵ�

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce(int*a, int*o, int n){
	__shared__ int sa[1024]; // �����޸�
	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	sa[tid] = a[idx];
	__syncthreads();

	for(int s = 1; s < blockDim.x; s *= 2){
		int index = tid*s*2;
		
		if(index < blockDim.x)
		{
			if(sa[index] < sa[index + s]) sa[index] = sa[index];
			else						  sa[index] = sa[index + s];
		}
		__syncthreads();
	}

	if(tid == 0) o[blockIdx.x] = sa[tid]; 
}
int main(){
	
	const int N			= 1000*1024;
	int block_size		= 1024;
	int block_num		= (N + block_size -1)/block_size;

	int *a, *o;
	a = (int *)malloc(sizeof(int) * N);
	o = (int *)malloc(sizeof(int) * block_num);
	
	for(int i = 0 ; i < N ; i++){
		a[i] = (i%2 == 0)? i : -i;
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
	reduce<<<block_num, block_size>>>(id,od,N);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end);
	
	printf("\n>>gpu_time = %f\n", gpu_time);
	
	cudaMemcpy(o,od,sizeof(int)*block_num, cudaMemcpyDeviceToHost);
	
	int min = INT_MAX;
	for(int i = 0 ; i < block_num ; i++){
		if(min > o[i]) min = o[i];
	}
	printf("gpu �ּҰ� = %d\n", min);
	
	/*for(int i = 0 ; i < block_num ; i ++){
		printf("%d ", o[i]);
	}*/
	printf("\n");

	//----------------------------------------------
	// cpu���� �ּҰ� ���ϱ�
	int min_cpu = INT_MAX;
	for(int i = 0 ; i < N ; i ++)
	{
		if(a[i] < min_cpu) 
			min_cpu = a[i];
	}
	
	printf("cpu �ּҰ� = %d", min_cpu);

	free(a); free(o);
	cudaFree(id); cudaFree(od);

	return 0;
}