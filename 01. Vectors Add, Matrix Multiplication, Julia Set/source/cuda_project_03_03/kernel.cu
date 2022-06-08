// 3�� ����-GPU�� CPU ��
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "opencv\cv.h"
#include "opencv\highgui.h"

using namespace cv;

typedef struct CudaComplex{ // ���Ҽ� ���� // ���� 3��
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

	// ������ �������ֱ� ���� Mid_x, Mid_y
	const int Mid_x = width / 2;
	const int Mid_y = height / 2;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(col >= width) return;
	if(row >= height) return;


	Cuda_C Complex_Result, Complex;

	// ��ȣ �� �����ϱ�
	Complex.r = (col - Mid_x) / 1024.0;
	Complex.i = (Mid_y - row) / 1024.0; 

	matrix[row * width + col] = 255;
	// ���� �ٸ��� �¿� ���Ѵٰ� ����

	for(int k = 0 ; k <= 200 ; k++){ // �ݺ����� ���鼭
		//Complex_Result = Complex
		if((Complex.r * Complex.r + Complex.i * Complex.i) > 1000000)
			// ���� �ݺ����� ���鼭 1000000 �� �Ѿ�� �߻��Ѵٰ� ����
		{
			matrix[row * width + col] = 0; break; // �Ѿ�� �� 0�� �����ϰ�, break
		}
		Complex_Result = complex_power(Complex);
		Complex_Result = complex_add((Complex_Result),c); 
		Complex = Complex_Result;
	}
}

int main(){
	//�ǽ� �߰� (CPU GPU ���Ѱ�)
	const int M = 2048; // x�� ����(width)
	const int N = 2048; // y�� ����(height)

	const int MidSpot_X = N/2; // x��ǥ ����
	const int MidSpot_Y = M/2; // y��ǥ ����
	
	// global �Լ��� ȣ���� ����� temp�� �����ϱ� ����
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
	
	printf("���Ҽ� C�� �Ǽ���, ����θ� �����Ͻÿ�(r,i) : ");
	scanf("%f,%f", &C.r, &C.i);
	printf("thread�� ������ �Է��ϼ��� : ");
	scanf("%d", &thread);
	

	
	mtGray.create(M,N,CV_8UC1);
	mtGray_cuda.create(M,N,CV_8UC1);
	
	//
	//Mat mtGray = Mat( M, N, CV_8UC1, Scalar(0));//CPU �׷��� �̹���
	//Mat mtGray_cuda = Mat( M, N, CV_8UC1, Scalar(0));//GPU �׷��� �̹���

	//gpu_time = 0;
	//cudaEventRecord(start, 0);
	dim3 dim_blocks(thread, thread, 1);
	dim3 dim_grid(N/thread, M/thread, 1);
	// dim_grid(x��, y��, z��)
	
	gpu_julia <<< dim_grid, dim_blocks>>>(dMat,M,N,C);
	//cudaEventRecord(end, 0);
	//cudaEventSynchronize(end);
	//cudaEventElapsedTime(&gpu_time, start, end);
	// printf("%f\n", gpu_time);

	cudaMemcpy(temp, dMat, sizeof(unsigned char)*M*N,cudaMemcpyDeviceToHost);
	//temp�� �����ϴ� ������ gpu ���α׷����� ���� dMat�� �ޱ� ����

	for(int i = 0 ; i < M; i ++){
		for(int j = 0 ; j < N ; j++){
			//printf("%d",temp[i*N + j]);
			mtGray_cuda.at<unsigned char>(i,j) = temp[i*N + j];
		}   
	}
	
	//----------------------------------------------------------------
	// CPU ���� ���
	for(int i = 0 ; i < M; i ++){
		for(int j = 0 ; j < N ; j++){
			Complex.r = (j - MidSpot_X) / 1024.0;   //x��
			Complex.i = (MidSpot_Y - i) / 1024.0;   //y��

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
	cv::waitKey(0); // â ������ ����!

	cv::imwrite("copy_CPU.jpg", mtGray);
	cv::imwrite("copy_CUDA.jpg", mtGray_cuda);


	free(temp);
	cudaFree(dMat);

	return 0;
}



