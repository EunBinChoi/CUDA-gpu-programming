// 3번 문제-GPU와 CPU 비교
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

using namespace cv;

typedef struct CudaComplex{ // 복소수 정의 // 문제 3번
	float r;
	float i;

}Cuda_C;

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

__global__ void gpu_julia(unsigned char* matrix, int width, int height, Cuda_C c){

	// 원점을 정의해주기 위한 Mid_x, Mid_y
	const int Mid_x = width / 2;
	const int Mid_y = height / 2;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(col >= width) return;
	if(row >= height) return;


	Cuda_C Complex_Result, Complex;

	// 부호 잘 생각하기
	Complex.r = (col - Mid_x) / 1024.0;
	Complex.i = (Mid_y - row) / 1024.0; 

	matrix[row * width + col] = 255;
	// 먼저 줄리아 셋에 속한다고 가정

	for(int k = 0 ; k <= 200 ; k++){ // 반복문을 돌면서
		//Complex_Result = Complex
		if((Complex.r * Complex.r + Complex.i * Complex.i) > 1000000)
			// 만약 반복문을 돌면서 1000000 이 넘어가면 발산한다고 가정
		{
			matrix[row * width + col] = 0; break; // 넘어갔을 때 0을 대입하고, break
		}
		Complex_Result = complex_power(Complex);
		Complex_Result = complex_add((Complex_Result),c); 
		Complex = Complex_Result;
	}
}

int main(){
	//실습 추가 (CPU GPU 비교한것)
	const int M = 2048; // x의 길이(width)
	const int N = 2048; // y의 길이(height)

	const int MidSpot_X = N/2; // x좌표 원점
	const int MidSpot_Y = M/2; // y좌표 원점
	
	// global 함수를 호출한 결과를 temp에 저장하기 위함
	unsigned char *temp;
	temp = (unsigned char *)malloc(sizeof(unsigned char) * M*N);
	
	for(int i = 0; i < M*N ; i++) temp[i] = 255;
	
	cv::Mat mtGray;
	cv::Mat mtGray_cuda;
	
	unsigned char *dMat;
	cudaMalloc((void**)&dMat, sizeof(unsigned char) * M*N);
	cudaMemcpy(dMat, temp, sizeof(unsigned char)*M*N,cudaMemcpyHostToDevice);

	int thread;
	
	Cuda_C Complex_Result, Complex, C;
	
	printf("복소수 C의 실수부, 허수부를 대입하시오(r,i) : ");
	scanf("%f,%f", &C.r, &C.i);
	printf("thread의 개수를 입력하세요 : ");
	scanf("%d", &thread);
	

	
	mtGray.create(M,N,CV_8UC1);
	mtGray_cuda.create(M,N,CV_8UC1);
	
	//
	//Mat mtGray = Mat( M, N, CV_8UC1, Scalar(0));//CPU 그레이 이미지
	//Mat mtGray_cuda = Mat( M, N, CV_8UC1, Scalar(0));//GPU 그레이 이미지

	//gpu_time = 0;
	//cudaEventRecord(start, 0);
	dim3 dim_blocks(thread, thread, 1);
	dim3 dim_grid(N/thread, M/thread, 1);
	// dim_grid(x축, y축, z축)
	
	gpu_julia <<< dim_grid, dim_blocks>>>(dMat,M,N,C);
	//cudaEventRecord(end, 0);
	//cudaEventSynchronize(end);
	//cudaEventElapsedTime(&gpu_time, start, end);
	// printf("%f\n", gpu_time);

	cudaMemcpy(temp, dMat, sizeof(unsigned char)*M*N,cudaMemcpyDeviceToHost);
	//temp을 설정하는 이유는 gpu 프로그램에서 계산된 dMat을 받기 위함

	for(int i = 0 ; i < M; i ++){
		for(int j = 0 ; j < N ; j++){
			//printf("%d",temp[i*N + j]);
			mtGray_cuda.at<unsigned char>(i,j) = temp[i*N + j];
		}   
	}
	
	//----------------------------------------------------------------
	// CPU 에서 계산
	for(int i = 0 ; i < M; i ++){
		for(int j = 0 ; j < N ; j++){
			Complex.r = (j - MidSpot_X) / 1024.0;   //x축
			Complex.i = (MidSpot_Y - i) / 1024.0;   //y축

			mtGray.at<unsigned char>(i,j) = 255;

			for(int k = 0 ; k <= 200 ; k++){
				//Complex_Result = Complex;

				if((Complex.r*Complex.r + Complex.i * Complex.i) > 1000000)
				{

					mtGray.at<unsigned char>(i,j) = 0;
					break;
				}
				else{
					Complex_Result = complex_power(Complex);
					Complex_Result = complex_add((Complex_Result),C); 
					Complex = Complex_Result;
				}
			}
		}
	}

	cv::imshow("Window_CPU", mtGray);
	cv::imshow("Window_CUDA", mtGray_cuda);
	cv::waitKey(0); // 창 여러개 띄우기!

	cv::imwrite("copy_CPU.jpg", mtGray);
	cv::imwrite("copy_CUDA.jpg", mtGray_cuda);


	free(temp);
	cudaFree(dMat);

	return 0;
}



