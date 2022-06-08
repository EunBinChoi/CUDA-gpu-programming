// 프린터 예제

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void filter(unsigned char*in, unsigned char*out, float *h, int num_rows, int num_cols){
	int xid = blockIdx.x * blockDim.x  + threadIdx.x;
	int yid = blockIdx.y * blockDim.y  + threadIdx.y;
	if(xid >= num_cols) return;
	if(yid >= num_rows) return;
	int idx = yid * num_cols + xid;
	
	float sum = 0;
	sum += h[0] * in[(yid-1)*num_cols + xid-1];
	sum += h[1] * in[(yid-1)*num_cols + xid  ];
	sum += h[2] * in[(yid-1)*num_cols + xid+1];
	sum += h[3] * in[(yid  )*num_cols + xid-1];
	sum += h[4] * in[(yid  )*num_cols + xid  ];
	sum += h[5] * in[(yid  )*num_cols + xid+1];
	sum += h[6] * in[(yid+1)*num_cols + xid-1];
	sum += h[7] * in[(yid+1)*num_cols + xid  ];
	sum += h[8] * in[(yid+1)*num_cols + xid+1];
	
	if(sum >= 255) out[idx] = 255;
	else if(sum <= 0) out[idx] = 0;
	else out[idx] = (unsigned char)sum;
}

int main()
{
	cv::Mat outGrey;
	cv::Mat inGrey = cv::imread("lenaTest.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	outGrey.create(inGrey.rows, inGrey.cols, CV_8UC1);
	
	unsigned char *h_inGrey = (unsigned char *)inGrey.ptr<unsigned char>(0);
	unsigned char *h_outGrey = (unsigned char *)outGrey.ptr<unsigned char>(0);

	unsigned char *d_in, *d_out;
	float *d_h;
	
	float h[][3] = {{1/9.,1/9.,1/9.},
				{1/9.,1/9.,1/9.},
				{1/9.,1/9.,1/9.}
	};
	
	int num_rows = inGrey.rows;
	int num_cols = inGrey.cols;
	int img_size = num_rows * num_cols;

	cudaMalloc((void **)&d_in, sizeof(unsigned char)*img_size);
	cudaMalloc((void **)&d_out, sizeof(unsigned char)*img_size);
	cudaMalloc((void **)&d_h, sizeof(float)*9);
	
	cudaMemcpy(d_in, h_inGrey, sizeof(unsigned char)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, h, sizeof(float)*9, cudaMemcpyHostToDevice);
	
	dim3 dim_block(32, 32, 1);
	dim3 dim_grid((num_cols + 31)/32, (num_rows + 31)/32,1);
	filter<<<dim_grid, dim_block>>>(d_in, d_out, d_h, num_rows, num_cols);
	
	cudaMemcpy(h_outGrey, d_out, sizeof(unsigned char)*img_size, cudaMemcpyDeviceToHost);
	cv::imwrite("lena_filterd.jpg", outGrey);
    return 0;
}


