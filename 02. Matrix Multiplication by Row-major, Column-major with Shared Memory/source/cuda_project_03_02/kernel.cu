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

#define TILE_DIM 32
//#define BLOCK_ROWS 16
#define BLOCK_COLS 32 // 여기서 block_cols은 tile_dim 과 같다고 가정

__global__ void gpu_transpose(int*a, int*b, int M, int N){
   __shared__ int tile[TILE_DIM][TILE_DIM];
   __shared__ int tile_result[TILE_DIM][TILE_DIM];

   int x = blockIdx.x * TILE_DIM + threadIdx.x;
   int y = blockIdx.y * TILE_DIM + threadIdx.y;
   
   int width = gridDim.x * TILE_DIM;
   int height = gridDim.y * TILE_DIM;

   //// 내가 짠 소스 (버전1) // 이 버전은 block_cols가 tile_dim보다 작아도 가능함(동시에 실행되는 쓰레드 cols의 갯수 < 타일 갯수)
   //for (int j = 0; j < TILE_DIM ; j += BLOCK_COLS) // 다음 행으로 넘어가기 위함
   //   tile[threadIdx.y][threadIdx.x + j] = a[(y * width) + x + j];

   //__syncthreads();
   //      
   //x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
   //y = blockIdx.x * TILE_DIM + threadIdx.y;

   //for (int j = 0; j < TILE_DIM; j += BLOCK_COLS){
   //   b[(y * height) + x + j] = tile[threadIdx.x + j][threadIdx.y];
   //}
   //__syncthreads();

   // 교수님께서 짠 소스 (버전2)
   //for (int j = 0; j < TILE_DIM ; j += BLOCK_COLS)
   
   tile[threadIdx.y][threadIdx.x] = a[(y*width) + x];

   __syncthreads();
         
   x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
   y = blockIdx.x * TILE_DIM + threadIdx.y;

   //for (int j = 0; j < TILE_DIM; j += BLOCK_COLS){
      b[(y*height) + x] = tile[threadIdx.x][threadIdx.y];
  // }
   __syncthreads();
}

int main(){
   cudaEvent_t start, end;
   float gpu_time; 

   cudaEventCreate(&start);
   cudaEventCreate(&end);

	const int M = 2048;
	const int N = 1024;
   //const int K = 1024;

   int *shared_a;
   int *shared_b;
   shared_a = (int *)malloc(sizeof(int) * M*N);
   shared_b = (int *)malloc(sizeof(int) * M*N);

   for(int i =0 ; i < M*N; i++) shared_a[i] = (int)(rand()%100);
   for(int i =0 ; i < M*N; i++) shared_b[i] = 0;

   int *shared_da;
   int *shared_db;

   cudaMalloc((void**)&shared_da, sizeof(int) * M*N);
   cudaMalloc((void**)&shared_db, sizeof(int) * M*N);
   cudaMemcpy(shared_da, shared_a, sizeof(int)*M*N,cudaMemcpyHostToDevice);

   int thread = 32;

   cudaEventRecord(start, 0);
   // 버전 1
   //dim3 dim_block(BLOCK_COLS, thread , 1);
   // 버전 2
   dim3 dim_block(thread,thread , 1);
   dim3 dim_grid(N/thread, M/thread, 1);
   //mat_mul<<<dim_grid, dim_block>>>(da,db,dc,M,N,K);
   gpu_transpose<<<dim_grid, dim_block>>>(shared_da,shared_db,M,N);

   cudaEventRecord(end, 0);
   cudaEventSynchronize(end);
   cudaEventElapsedTime(&gpu_time, start, end); // 시간 간격을 잼

   //cudaMemcpy(c, dc, sizeof(float)*M*N,cudaMemcpyDeviceToHost);
   cudaMemcpy(shared_b, shared_db, sizeof(int)*M*N,cudaMemcpyDeviceToHost);

   printf("블럭당 thread 개수 = %d, shared gpu time = %f\n", thread, gpu_time);

   //for(int i = 0 ; i < M ; i++){
   //   for(int j = 0 ; j < N ; j++)
   //      printf("%10f ", shared_c[i*N+j]);
   //   printf("\n");
   //}
   int i = 0;
   int j = 0;

   //for(i = 0 ; i < M ; i++){
   //   for(j = 0; j < N ; j++){
   //      printf("%4d ",shared_a[i*N+j]);        
   //   }
   //   printf("\n");
   //}
   //printf("\n");printf("\n");
   //
   //for(i = 0 ; i < N ; i++){
   //   for(j = 0; j < M ; j++){
   //      printf("%4d ",shared_b[i*M+j]);        
   //   }
   //   printf("\n");
   //}
   // 
   printf("\n");
   for(i = 0 ; i < M ; i++){
      for(j = 0; j < N ; j++){
         // printf("%4d ",shared_a[i*N+j]);
         if(abs(shared_a[i*N+j]-shared_b[j*M+i]) > 0.0001){
            printf("=>[ false ]\n");
            //  printf("%d %d %d %d\n",i,j ,shared_a[i*N+j],shared_b[j*M+i]);
            return 0;
         }
      }
      //printf("\n");
   }
   if(i*j == M*N) printf("=>[ 모두 일치합니다 ]\n");
  
   
   free(shared_a); free(shared_b);
   cudaFree(shared_da); cudaFree(shared_db);
   
   return 0;
}