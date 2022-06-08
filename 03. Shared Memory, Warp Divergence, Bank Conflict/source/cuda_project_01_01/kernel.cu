
// reduce 0 ���α׷� ���� 
// reduce 0 => �����޸𸮸� ������� �ʴ� ����
// reduce 1 => �����޸� ������
// reduce 2 => ���� �б� �ذ�
// reduce 3 => ��ũ �浹 �ذ�

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce(int*a, int*o, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	for(int s = 1; s < blockDim.x; s *= 2){ // �� �� ���� �������� �ݺ�
		if(idx % (2*s) == 0)
		{
			// ���� �ִ� ���� ���Ͽ� �ּҰ��� ����
			if(a[idx] < a[idx + s])  
				a[idx] = a[idx];
			else
				a[idx] = a[idx + s];
		} 
		__syncthreads();
	}

	if(threadIdx.x == 0) o[blockIdx.x] = a[idx];
}
int main(){
	const int N		= 1000 * 1024;
	int	block_size	= 1024;
	int	block_num	= (N + block_size -1) / block_size;

	int *a, *o;
	a = (int *)malloc(sizeof(int) * N);
	o = (int *)malloc(sizeof(int) * block_num);
	
	for(int i = 0 ; i < N ; i++)
	{
		a[i] = (i % 2 == 0)? i : -i;
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
	
	printf("\n");
	printf("gpu_time = %f\n", gpu_time);

	cudaMemcpy(o,od,sizeof(int)*block_num, cudaMemcpyDeviceToHost);

	// block �� ���� �ּҰ��� �߿��� �ּҰ��� ����
	int min = INT_MAX;
	for(int i = 0 ; i < block_num ; i ++){
		if(o[i] < min) min = o[i];
	}

	printf("\nGPU �ּҰ� = %d\n", min);
	

	// ---------------------------------------
	// CPU ���� �ּҰ�
	int min_cpu = INT_MAX;
	for(int i = 0 ; i < N ; i ++)
	{
		if(a[i] < min_cpu) 
			min_cpu = a[i];
	}
	
	printf("CPU �ּҰ� = %d\n", min_cpu);
	
	free(a); free(o);
	cudaFree(id); cudaFree(od);

	return 0;
}