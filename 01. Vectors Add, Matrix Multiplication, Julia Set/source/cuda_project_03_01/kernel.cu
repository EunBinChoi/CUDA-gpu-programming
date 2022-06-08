
//3번문제-GPU만 관련된 것

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

//#define NUM_DATA 123456
using namespace cv;

typedef struct CudaComplex{ // 복소수 정의 // 문제 3번
	float r;
	float i;

}Cuda_C;

__device__ __host__ void complex_add(float a, float b, float c, float d, float *realOut, float *imgOut){
	*realOut = a + c;
	*imgOut = b + d;

}

__device__ __host__ void complex_sub(float a, float b, float c, float d, float *realOut, float *imgOut){
	*realOut = a - c;
	*imgOut = b - d;

}

__device__ __host__ void complex_mul(float a, float b, float c, float d, float *realOut, float *imgOut){
	*realOut = (a*c) - (b*d);
	*imgOut = (b*c) + (a*d);

}

__device__ __host__ void complex_div(float a, float b, float c, float d, float *realOut, float *imgOut){
	*realOut = ((a*c) + (b*d)) / (pow(c,2)+pow(d,2));
	*imgOut = ((b*c) - (a*d)) / (pow(c,2)+pow(d,2));

}
__device__ __host__ Cuda_C complex_add(Cuda_C a, Cuda_C b){

	Cuda_C result;
	result.r = a.r + b.r;
	result.i = a.i + b.i;

	return result;

}
__device__ __host__ Cuda_C complex_mul(Cuda_C a, Cuda_C b){

	Cuda_C result;
	result.r = (a.r*b.r) - (a.i*b.i);
	result.i = (a.r*b.i) + (a.i*b.r);

	return result;
}
__device__ __host__ Cuda_C complex_power(Cuda_C a){

	Cuda_C result;
	result = complex_mul(a,a);
	return result;
}


__global__ void vec_add(float *a, float *b, float *c, int n){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= n) return;
	c[tid] = a[tid] + b[tid];

}

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

__global__ void gpu_julia(unsigned char* matrix, int width, int height, Cuda_C c){

	const int Mid_x = width / 2;
	const int Mid_y = height / 2;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	Cuda_C Complex_Result;
	Cuda_C Complex;

	Complex.r = (col - Mid_x) / 1024.0;
	Complex.i = (Mid_y - row) / 1024.0; 

	matrix[row*width + col] = 255;

	for(int k = 0 ; k <= 200 ; k++){

		Complex_Result = complex_power(Complex);
		if((Complex_Result.r*Complex_Result.r + Complex_Result.i * Complex_Result.i) > 1000000)
		{
			matrix[row*width + col] = 0; break;
		}

		Complex_Result = complex_add((Complex_Result),c); 
		Complex = Complex_Result;
	}
}

int main(){
	// 실습 3-1
   Cuda_C C;
   printf("복소수 C의 실수부, 허수부를 대입하시오(r,i) : ");
   scanf("%f,%f", &C.r, &C.i);

   const int M = 2048; // x의 길이(width)
   const int N = 2048; // y의 길이(height)

   ////matrix = (float *)malloc(sizeof(float) *M*N);
   unsigned char *dMat;
   cudaMalloc((void**)&dMat, sizeof(char) *M*N);
   int thread;
   printf("thread의 개수를 입력하시오 : ");
   scanf("%d", &thread);

   dim3 dim_blocks(thread, thread, 1);
   dim3 dim_grid(N/thread, M/thread, 1);

   Mat mtGray = Mat( M, N, CV_8UC1, Scalar(0));
   gpu_julia <<< dim_grid, dim_blocks >>> (dMat,M,N,C);
   unsigned char *tmp;
   tmp = (unsigned char *)malloc(sizeof(char) * M*N);
   cudaMemcpy(tmp, dMat, sizeof(char)*M*N,cudaMemcpyDeviceToHost);
   
   for(int i = 0 ; i < M; i ++){
      for(int j = 0 ; j < N ; j++){

         if(tmp[i*M + j] == 255) mtGray.at<unsigned char>(i,j) = 255; 
         else if(tmp[i*M + j] == 0) mtGray.at<unsigned char>(i,j) = 0;


      }
   }

   imshow("Window", mtGray);
   waitKey(0); //사용자가 키를 누르기 전까지 기다림
   imwrite("copy.jpg",mtGray);
   //이미지 픽셀이 너무 커서 저장함
   
   free(tmp);
   cudaFree(dMat);
   return 0;
}
 