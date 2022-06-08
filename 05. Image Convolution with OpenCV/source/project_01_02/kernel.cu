// OPENCV °úÁ¦

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void enlighten(uchar4 *in, uchar4 *out, int num_rows, int num_cols){
	int xid = blockIdx.x * blockDim.x  + threadIdx.x;
	int yid = blockIdx.y * blockDim.y  + threadIdx.y;
	if(xid >= num_cols) return;
	if(yid >= num_rows) return;
	int idx = yid * num_cols + xid;
	
	unsigned char red = in[idx].x;
	unsigned char green = in[idx].y;
	unsigned char blue = in[idx].z;

	out[idx] = make_uchar4(red,green,blue, 255);
}

int main()
{
	cv::Mat inRGBA;
	cv::Mat outRGBA;

	cv::Mat image = cv::imread("dogTest.jpg", CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(image, inRGBA, CV_BGR2RGBA);
	outRGBA.create(image.rows, image.cols, CV_8UC4);
	
	int num_rows = inRGBA.rows;
	int num_cols = inRGBA.cols;
	int img_size = num_rows * num_cols;

	uchar4	*h_inRGBA  = (uchar4 *)inRGBA.ptr<unsigned char>(0);
	uchar4  *h_outRGBA = (uchar4 *)outRGBA.ptr<unsigned char>(0);

	uchar4 *d_in, *d_out;

	cudaMalloc((void **)&d_in, sizeof(uchar4)*img_size);
	cudaMalloc((void **)&d_out, sizeof(uchar4)*img_size);
	cudaMemcpy(d_in, h_inRGBA, sizeof(uchar4)*img_size, cudaMemcpyHostToDevice);
	
	dim3 dim_block(32, 32, 1);
	dim3 dim_grid((num_cols + 31)/32, (num_rows + 31)/32,1);
	
	enlighten<<<dim_grid, dim_block>>>(d_in, d_out, num_rows, num_cols);
	
	cudaMemcpy(h_outRGBA, d_out, sizeof(uchar4)*img_size, cudaMemcpyDeviceToHost);
	
	cv::Mat imageOutBGR;
	cv::cvtColor(outRGBA, imageOutBGR, CV_RGBA2BGR);
	cv::imwrite("dog_filterd.jpg", imageOutBGR);

	cudaError_t err;
	err = cudaGetLastError();
	printf("%s\n", cudaGetErrorString(err));

	
	cudaFree(d_in);
	cudaFree(d_out);

    return 0;
}


