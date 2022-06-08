
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<time.h>
#include<stdlib.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\opencv.hpp>

struct Complex
{
   double real;
   double imagine;
};

__global__ void enlighten(uchar4 *out, int num_rows, int num_cols, int N, Complex C) { //kernel
   int xid = blockIdx.x * blockDim.x + threadIdx.x;
   int yid = blockIdx.y * blockDim.y + threadIdx.y;

   float real, imagine;
   float next_real, next_imagine;
   bool infinite = 0;
   int i;

   if (xid >= num_cols) return;
   if (yid >= num_rows) return;
   int idx = yid * num_cols + xid;

   real = (xid - 1024.0) / 1024.0;
   imagine = (1024.0 - yid) / 1024.0;

   for (i = 0; i<200; i++) {
      if (real * real + imagine * imagine > 1e6) {
         infinite = true;
         break;
      }
      next_real = real * real - imagine * imagine + C.real;
      next_imagine = 2 * real * imagine + C.imagine;

      real = next_real;
      imagine = next_imagine;
   }


   if (infinite == true)
      out[idx] = make_uchar4(255, 255, 255, 255);
   else {
      out[idx] = make_uchar4(0, 0, 0, 255);
   }

}

int main(void) {
   // C값 받기
   Complex C = {};

   printf("C의 실수 부 입력 : ");
   scanf("%lf", &C.real);
   printf("C의 허수 부 입력 : ");
   scanf("%lf", &C.imagine);

   const int M = 2048;
   const int N = 2048;

   cv::Mat outRGBA;
   outRGBA.create(M, N, CV_8UC4);

   int num_rows = outRGBA.rows;
   int num_cols = outRGBA.cols;
   int img_size = M * N;


   uchar4 *h_outRGBA = (uchar4 *)outRGBA.ptr<unsigned char>(0);
   uchar4 *d_out;
   cudaMalloc((void **)&d_out, sizeof(uchar4) * img_size);


   dim3 dim_block(32, 32, 1);
   dim3 dim_grid((num_cols + 31) / 32, (num_rows + 31) / 32, 1);   //64, 64
   enlighten << <dim_grid, dim_block >> > (d_out, num_rows, num_cols, N, C);

   cudaMemcpy(h_outRGBA, d_out, sizeof(uchar4)*img_size, cudaMemcpyDeviceToHost);
   cv::Mat imageOutBGR;
   cv::cvtColor(outRGBA, imageOutBGR, CV_RGBA2BGR);
   cv::imwrite("testout.jpg", imageOutBGR);

   cudaFree(d_out);

}