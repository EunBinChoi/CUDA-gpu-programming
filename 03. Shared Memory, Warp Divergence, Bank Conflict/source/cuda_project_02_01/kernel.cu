// reduce 0
// 공유 메모리 없이 하는 것

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void reduce(int *a, int *b, int *c, int *o, int n){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
   
   // 해결
   if(idx >= n) return;

   // 왜 배열이 c일때 안되는지
   c[idx] = (a[idx] * b[idx]);
   __syncthreads();

   for(int s = 1; s < blockDim.x ; s *= 2)
   {   
       if(idx % (2*s) == 0)
	   {
          c[idx] = c[idx] + c[idx + s];
	   }
      __syncthreads();
   }
   
   if(tid == 0) o[blockIdx.x] = c[idx];
}

int main(){
	// 파일 관련
	FILE *fp1, *fp2;

	char file_name1[50] = "seta.dat";
	char file_name2[50] = "setb.dat";
	
	int *matrix_a ,int *matrix_b;

	fp1 = fopen(file_name1, "r");
	if(fp1 == NULL){
		printf("File Open Error");
		return;
	}

	int num1;
	fscanf(fp1, "%d", &num1);
	
	fp2 = fopen(file_name2, "r");
	if(fp2 == NULL){
		printf("File Open Error");
		return;
	}
	
	int num2;
	fscanf(fp2, "%d", &num2);
	
	matrix_a = (int *)malloc(sizeof(int) * num1);
	matrix_b = (int *)malloc(sizeof(int) * num2);

	for(int i = 0 ; i < num1 ; i++){
		fscanf(fp1, "%d", &(matrix_a[i]));
	}

	for(int i = 0 ; i < num2 ; i++){
		fscanf(fp2, "%d",&(matrix_b[i]));
	}
	printf("\n");
	
	fclose(fp1);
	fclose(fp2);
	
		
	// 벡터 관련
	//const int N = 1000 * 1024; // N은 저장되어있으므로 따로 정의해줄 필요 없음
	int block_size = 1024;
	// 블럭 사이즈는 2 이상이어야한다
	int block_num = (num1 + block_size -1) / block_size;
	int *o = (int *)malloc(sizeof(int) * block_num);

	int *da, *db, *dc, *dd;
	cudaMalloc((void **)&da, sizeof(int)*num1);
	cudaMalloc((void **)&db, sizeof(int)*num1);
	cudaMalloc((void **)&dc, sizeof(int)*num1);
	cudaMalloc((void **)&dd, sizeof(int)*block_num);
	
	cudaEvent_t start, end;
	float gpu_time;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaMemcpy(da,matrix_a,sizeof(int)*num1, cudaMemcpyHostToDevice);
	cudaMemcpy(db,matrix_b,sizeof(int)*num1, cudaMemcpyHostToDevice);
	// c배열은 가져올 필요 x

	cudaEventRecord(start, 0);
	reduce<<<block_num, block_size>>>(da,db,dc,dd,num1);
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&gpu_time, start, end);
	
	cudaMemcpy(o,dd,sizeof(int)*block_num, cudaMemcpyDeviceToHost);
	

	printf("\n\n");
	printf("<< 결과값 >> \n");
	int sum = 0;
	for(int i = 0 ; i < block_num ; i ++){
		sum += o[i];
	}
	printf("\n\n");
	printf(">> sum : %d", sum);
	printf("\n\n");
	printf("<< 실행시간 >>\ngpu_time = %f\n", gpu_time);

	// ---------------------
	// CPU에서 
	int result = 0;
	for(int i = 0; i < num1; i++){
		result += matrix_a[i] * matrix_b[i];
	}
	printf("\n\n");
	printf(">> sum : %d\n", result);
	return 0;
}