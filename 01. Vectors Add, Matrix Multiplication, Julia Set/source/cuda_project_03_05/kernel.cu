#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WIDTH 2048
#define HEIGHT 2048
#define CENTER_X 1024
#define CENTER_Y 1024

__global__ void julia_set(int *result, double cr, double ci) {
   int xid = blockIdx.x * blockDim.x + threadIdx.x;
   int yid = blockIdx.y * blockDim.y + threadIdx.y;

   if (xid >= WIDTH) return;
   if (yid >= HEIGHT) return;
   int idx = yid * WIDTH + xid;

   double zr = (long double)(xid - CENTER_X) / CENTER_X;
   double zi = (long double)(-yid + CENTER_Y) / CENTER_Y;
   double temp_real;
   double temp_imag;

   for (int i = 0 ; i < 200 ; i++) {
      temp_real = zr;
      temp_imag = zi;
      zr = temp_real * temp_real - temp_imag * temp_imag + cr;
      zi = 2 * temp_real * temp_imag + ci;
      if (zr * zr + zi * zi > 1000000) {
         result[idx] = 1;
         return;
      }
   }
   result[idx] = 0;
}

__global__ void draw(uchar4 *out, int *result) {

   int xid = blockIdx.x * blockDim.x + threadIdx.x;
   int yid = blockIdx.y * blockDim.y + threadIdx.y;
   if (xid >= WIDTH) return;
   if (yid >= HEIGHT) return;
   int idx = yid * WIDTH + xid;

   if (result[idx])
      out[idx] = make_uchar4(255, 255, 255, 255);
   else
      out[idx] = make_uchar4(0, 0, 0, 255);
   //rgb[idx] = make_uchar4(result[idx], result[idx]<<3, result[idx] << 2, 255);
}

int main() {
   cv::Mat outRGBA;
   outRGBA.create(HEIGHT, WIDTH, CV_8UC4);

   uchar4 *h_outRGBA;
   h_outRGBA = (uchar4*)outRGBA.ptr<unsigned char>(0);

   uchar4 *c_RGB;
   int *c_result;
   cudaMalloc((void**)&c_RGB, sizeof(uchar4) * HEIGHT * WIDTH);
   cudaMalloc((void**)&c_result, sizeof(int) * HEIGHT * WIDTH);
   
   dim3 dim_grid((WIDTH+31) / 32, (HEIGHT+31) / 32, 1);
   dim3 dim_block(32, 32, 1);

   julia_set <<<dim_grid, dim_block >>>(c_result, -0.8, 0.156);
   draw <<<dim_grid, dim_block >>>(c_RGB, c_result);

   cudaMemcpy(h_outRGBA, c_RGB, sizeof(uchar4) * HEIGHT * WIDTH, cudaMemcpyDeviceToHost);

   cv::Mat imageOutBGR;
   cv::cvtColor(outRGBA, imageOutBGR, CV_RGBA2BGR);
   cv::imwrite("output2.jpg", imageOutBGR);

   return 0;
} 