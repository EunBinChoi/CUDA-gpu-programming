#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

__global__ void sperate(uchar4 *in, unsigned char* outRed, unsigned char *outGreen, unsigned char *outBlue,
	int num_rows, int num_cols)
{
	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	int yid = blockIdx.y * blockDim.y + threadIdx.y;
	if (xid >= num_cols) return;
	if (yid >= num_rows) return;
	int idx = yid * num_cols + xid;

	outRed[idx] = in[idx].x;
	outGreen[idx] = in[idx].y;
	outBlue[idx] = in[idx].z;
}

__global__ void convolution(unsigned char* in, unsigned char* out, float *h, int num_rows, int num_cols) {

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	int yid = blockIdx.y * blockDim.y + threadIdx.y;
	if (xid >= num_cols) return;
	if (yid >= num_rows) return;
	int idx = yid * num_cols + xid;

	float sum = 0;
	
	((yid < 1 || xid < 1)				? sum += 0 : sum += h[0] * in[(yid - 1)*num_cols + xid - 1]);
	((yid < 1)							? sum += 0 : sum += h[1] * in[(yid - 1)*num_cols + xid]);
	((yid < 1 || xid + 1 >= num_cols)	? sum += 0 : sum += h[2] * in[(yid - 1)*num_cols + xid + 1]);
	((xid < 1)							? sum += 0 : sum += h[3] * in[(yid)*num_cols + xid - 1]);
	
	sum += h[4] * in[(yid)*num_cols + xid];
	
	((xid +1 >= num_cols)				? sum += 0 : sum += h[5] * in[(yid)*num_cols + xid + 1]);
	((yid +1 >= num_rows || xid < 1)	? sum += 0 : sum += h[6] * in[(yid + 1)*num_cols + xid - 1]);
	((yid +1 >= num_rows)				? sum += 0 : sum += h[7] * in[(yid + 1)*num_cols + xid]);
	((yid +1 >= num_rows || xid +1 >= num_cols) ? sum += 0 : sum += h[8] * in[(yid + 1)*num_cols + xid + 1]);

	if (sum >= 255) out[idx] = 255;
	else if (sum <= 0) out[idx] = 0;
	else out[idx] = (unsigned char)sum;
}

__global__ void gather(unsigned char *outRed, unsigned char *outGreen, unsigned char *outBlue,
	uchar4 *out, int num_rows, int num_cols)
{
	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	int yid = blockIdx.y * blockDim.y + threadIdx.y;
	if (xid >= num_cols) return;
	if (yid >= num_rows) return;
	int idx = yid * num_cols + xid;

	out[idx] = make_uchar4(outRed[idx], outGreen[idx], outBlue[idx], 255);
}

int main()
{
	cv::Mat inRGBA;
	cv::Mat outRGBA;

	cv::Mat image = cv::imread("dog.jpg", CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(image, inRGBA, CV_BGR2RGBA);
	// image(이미지)를 inRGBA(배열)에 저장
	outRGBA.create(image.rows, image.cols, CV_8UC4);

	int num_rows = inRGBA.rows;
	int num_cols = inRGBA.cols;
	int img_size = num_rows * num_cols;

	uchar4	*h_inRGBA = (uchar4 *)inRGBA.ptr<unsigned char>(0);
	uchar4	*h_outRGBA = (uchar4 *)outRGBA.ptr<unsigned char>(0);

	uchar4 *d_in, *d_out;
	unsigned char* d_outRed, *d_outGreen, *d_outBlue;
	//unsigned char* dd_outRed, *dd_outGreen, *dd_outBlue;

	float *d_h;
	float h[][3] = {{ 1/9.,1/9.,1/9. },
					{ 1/9.,1/9.,1/9. },
					{ 1/9.,1/9.,1/9. }};
	
	/*float h[][3] = {{ -1,-1,-1 },
					{ -1,9.,-1 },
					{ -1,-1,-1 }};
*/
	cudaMalloc((void **)&d_in, sizeof(uchar4)*img_size);
	cudaMalloc((void **)&d_out, sizeof(uchar4)*img_size);
	cudaMalloc((void **)&d_h, sizeof(float) * 9);

	cudaMalloc((void **)&d_outRed, sizeof(unsigned char)*img_size);
	cudaMalloc((void **)&d_outGreen, sizeof(unsigned char)*img_size);
	cudaMalloc((void **)&d_outBlue, sizeof(unsigned char)*img_size);

	cudaMemcpy(d_in, h_inRGBA, sizeof(uchar4)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_outRGBA, sizeof(uchar4)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, h, sizeof(float) * 9, cudaMemcpyHostToDevice);

	dim3 dim_block(32, 32, 1);
	dim3 dim_grid((num_cols + 31) / 32, (num_rows + 31) / 32, 1);

	sperate << <dim_grid, dim_block >> >(d_in, d_outRed, d_outGreen, d_outBlue, num_rows, num_cols);

	convolution << <dim_grid, dim_block >> >(d_outRed, d_outRed, d_h, num_rows, num_cols);
	convolution << <dim_grid, dim_block >> >(d_outGreen, d_outGreen, d_h, num_rows, num_cols);
	convolution << <dim_grid, dim_block >> >(d_outBlue, d_outBlue, d_h, num_rows, num_cols);

	gather << <dim_grid, dim_block >> >(d_outRed, d_outGreen, d_outBlue, d_out, num_rows, num_cols);
	cudaMemcpy(h_outRGBA, d_out, sizeof(uchar4)*img_size, cudaMemcpyDeviceToHost);

	cv::Mat imageOutBGR;
	cv::cvtColor(outRGBA, imageOutBGR, CV_RGBA2BGR);
	// outRGBA : 배열
	// imageOutBGR : 이미지
	cv::imwrite("out.jpg", imageOutBGR);

	return 0;
}


