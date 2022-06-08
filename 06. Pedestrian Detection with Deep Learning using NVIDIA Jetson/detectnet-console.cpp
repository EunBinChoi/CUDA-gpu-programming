/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"
#include "loadImage.h"

#include "cudaMappedMemory.h"

#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include "jetsonGPIO.h"

using namespace std;

// Draw Rectangle

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

// 이미지를 처리해주는 cpuRectOutlines 함수 정의
template <typename T> // template 을 이용 
void cpuRectOutlines(T* input, T* output, int width, int height, float4* rects, int numRects, float4 color ) 
{	

	T* tmp = (T*)malloc(sizeof(T)*width*height); 
	// tmp는 현재 이미지의 모든 픽셀을 의미하는 input의 임시변수로, 이를 동적할당하여 사용함
	
	// height * width 만큼 이중 반복문들 돌면서
	for(int i = 0 ; i < height ; i ++){ 
		for(int j = 0 ; j < width ; j++){
			tmp[i*width + j] = input[i*width + j];	// input 배열의 복사본인 tmp에 input의 i,j에 해당하는 값을 모두 복사
		}
	}

	// 이미지 중간에 이미지 절반 크기의 사각형 테두리를 그림
	for(int i = (int)((height)/4.) ; i <= (int)((height*3)/4.); i ++){
		for(int j = (int)((width)/4.); j <= (int)((width*3)/4.); j ++){
		/* 이미지 중간에 이미지 절반 크기의 사각형 테두리를 그리기 위해서는 
		가장 바깥의 반복문에서는 height*1/4~height*3/4 사이의 값을 반복하면서
		안쪽 반복문에서는 width*1/4~width*3/4 사이의 값을 반복하면서
		
		(이 사이값만 보는 이유는 이미지 절반 크기의 사각형 내부를 확인하면서 테두리에 해당하는 값이 있는지 확인하기 위함이다) 
		*/
			if((i == (int)((height)/4.)) || (i == (int)((height*3)/4.))){ // 만약 i가 테두리(height/4, height*3/4)에 해당한다면
				// 빨간색으로 칠함
				tmp[i*width + j].x = color.x;
				tmp[i*width + j].y = color.y;
				tmp[i*width + j].z = color.z;
				tmp[i*width + j].w = color.w;
			}

			if((j == (int)((width)/4.)) || (j == (int)((width*3)/4.))){ // 만약 j가 테두리(width/4, width*3/4)에 해당한다면
				// 빨간색으로 칠함
				tmp[i*width + j].x = color.x;
				tmp[i*width + j].y = color.y;
				tmp[i*width + j].z = color.z;
				tmp[i*width + j].w = color.w;
			}
			
		}
	}



	int count = 0; // 그림 중앙의 사각형 테두리에 들어가는 점의 갯수를 저장하는 변수

	for( int nr = 0; nr < numRects; nr++ ) // 사각형의 갯수만큼 반복문을 돌면서
	{
		const float4 r = rects[nr]; // r이라는 변수에 rects[nr]의 값을 대입
		//printf("=>%f %f %f %f", r.x,r.y,r.z,r.w);
	
		float xid = (r.x + r.z) / 2; // xid 를 이미지의 가로 중간값으로 정의
		float yid = (r.y + r.w) / 2; // yid 를 이미지의 세로 중간값으로 정의
		
		/*아래에서 구현한 방법은 보행자 bounding box에 가운데 점을 찍을 때, 한 픽셀로 찍으면 눈에 보이지 않으므로,
		가운데 점에 해당하는 픽셀과 주변의 점 총 9개의 점을 찍어 눈에 보일 수 있도록 하는 것이다.
		실제 bounding box의 가운데 점은 tmp[yid][xid]에 해당하는 것이고, 주변의 픽셀까지 빨간색으로 색칠해 눈에 잘 보일 수 있도록 하기 위함이다*/
		
		// tmp[yid-1][xid-1]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid-1 + ((int)yid-1) * width].x = color.x; 
		tmp[(int)xid-1 + ((int)yid-1) * width].y = color.y;
		tmp[(int)xid-1 + ((int)yid-1) * width].z = color.z;
		tmp[(int)xid-1 + ((int)yid-1) * width].w = color.w;	
		
				
		// tmp[yid][xid-1]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid-1 + ((int)yid) * width].x = color.x;
		tmp[(int)xid-1 + ((int)yid) * width].y = color.y;
		tmp[(int)xid-1 + ((int)yid) * width].z = color.z;
		tmp[(int)xid-1 + ((int)yid) * width].w = color.w;

		// tmp[yid+1][xid-1]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid-1 + ((int)yid+1) * width].x = color.x;
		tmp[(int)xid-1 + ((int)yid+1) * width].y = color.y;
		tmp[(int)xid-1 + ((int)yid+1) * width].z = color.z;
		tmp[(int)xid-1 + ((int)yid+1) * width].w = color.w;	

  		// tmp[yid-1][xid]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid + ((int)yid-1) * width].x = color.x;
		tmp[(int)xid + ((int)yid-1) * width].y = color.y;
		tmp[(int)xid + ((int)yid-1) * width].z = color.z;
		tmp[(int)xid + ((int)yid-1) * width].w = color.w;	
	
		 
		// tmp[yid][xid]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid + (int)yid * width].x = color.x;
		tmp[(int)xid + (int)yid * width].y = color.y;
		tmp[(int)xid + (int)yid * width].z = color.z;
		tmp[(int)xid + (int)yid * width].w = color.w;

		// tmp[yid+1][xid]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid  + ((int)yid+1) * width].x = color.x;
		tmp[(int)xid  + ((int)yid+1) * width].y = color.y;
		tmp[(int)xid  + ((int)yid+1) * width].z = color.z;
		tmp[(int)xid  + ((int)yid+1) * width].w = color.w;	

		// tmp[yid-1][xid+1]에 해당하는 픽셀을 빨간색으로 칠함
		input[(int)xid+1  + ((int)yid-1) * width].x = color.x;
		input[(int)xid+1  + ((int)yid-1) * width].y = color.y;
		input[(int)xid+1  + ((int)yid-1) * width].z = color.z;
		input[(int)xid+1  + ((int)yid-1) * width].w = color.w;

		// tmp[yid][xid+1]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid+1 + (int)yid * width].x = color.x;
		tmp[(int)xid+1 + (int)yid * width].y = color.y;
		tmp[(int)xid+1 + (int)yid * width].z = color.z;
		tmp[(int)xid+1 + (int)yid * width].w = color.w;

 		
		// tmp[yid+1][xid+1]에 해당하는 픽셀을 빨간색으로 칠함
		tmp[(int)xid+1 + ((int)yid+1) * width].x = color.x;
		tmp[(int)xid+1 + ((int)yid+1) * width].y = color.y;
		tmp[(int)xid+1 + ((int)yid+1) * width].z = color.z;
		tmp[(int)xid+1 + ((int)yid+1) * width].w = color.w;

		
		/*만약 그림 절반 크기 사각형 내부에 xid, yid 가 포함되어있다면
		
		여기서 xid가 사각형 내부에 포함되려면 width/4~width*3/4 사이
		yid는 height/4~height*3/4 사이 값이여야 함*/

		if( ((int)xid > (int)((width)/4.)) && ((int)xid < (int)((width)*3/4.)) ){
			if ( ((int)yid > (int)((height)/4.)) && ((int)yid < (int)((height)*3/4.)) )
				count ++; // 그림 중앙의 사각형 테두리에 들어가는 점의 갯수를 저장하는 변수인 count를 증가
		}

	}

	printf("\n\ncount : %d\n", count); // count를 출력

	jetsonTX1GPIONumber led[4] = {gpio37,gpio219,gpio36,gpio63}; // 각 led에 해당하는 gpio핀을 설정
	
	// 교수님이 주신 코드
	gpioExport(led[0]); 
	gpioExport(led[1]);	
	gpioExport(led[2]);	
	gpioExport(led[3]);	

	gpioSetDirection(led[0], outputPin);
	gpioSetDirection(led[1], outputPin);
	gpioSetDirection(led[2], outputPin);
	gpioSetDirection(led[3], outputPin);

	for(int i = 0;i < count;i++) { 
		// 중앙의 사각형에 들어가는 빨간점이 몇 개인지 저장하는 변수(count) 만큼 돌면서 led를 출력 
		gpioSetValue(led[i], on);
	}
	
	// height * width 만큼 이중 반복문을 돌면서
	for(int i = 0 ; i < height ; i ++){
		for(int j = 0 ; j < width ; j++){
			input[i*width + j] = tmp[i*width + j]; 
			// 이미지 처리에서 사용되었던 임시변수 tmp를 다시 input에 대입
			// 이미지 처리 중 변경된 값을 input에 적용시키기 위함
		}
	}
	
}

// main entry point
int main( int argc, char** argv )
{
	printf("detectnet-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	/*jetsonTX1GPIONumber	led0 = gpio37;
	jetsonTX1GPIONumber	led1 = gpio219;
	jetsonTX1GPIONumber	led2 = gpio36;
	jetsonTX1GPIONumber	led3 = gpio63;*/


	// retrieve filename argument
	if( argc < 2 )
	{
		printf("detectnet-console:   input image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	

	// create detectNet
	detectNet* net = detectNet::Create( detectNet::PEDNET_MULTI ); // uncomment to enable one of these 
  //detectNet* net = detectNet::Create( detectNet::PEDNET );
  //detectNet* net = detectNet::Create( detectNet::FACENET );
  //detectNet* net = detectNet::Create("multiped-500/deploy.prototxt", "multiped-500/snapshot_iter_178000.caffemodel", "multiped-500/mean.binaryproto" );
	
	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
		return 0;
	}

	net->EnableProfiler();
	
	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		
	printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	
	float* confCPU  = NULL;	
	float* confCUDA = NULL;
	
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)))
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	
	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if(!loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}


	int numBoundingBoxes = maxBoxes;
	
	printf("detectnet-console:  beginning processing network (%zu)\n", current_timestamp());

	const bool result = net->Detect(imgCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU);

	printf("detectnet-console:  finished processing network  (%zu)\n", current_timestamp());
	
	
	if( !result )
		printf("detectnet-console:  failed to classify '%s'\n", imgFilename);
	
	else if( argc > 2 )		// if the user supplied an output filename
	{
		printf("%i bounding boxes detected\n", numBoundingBoxes);

		int lastClass = 0;
		int lastStart = 0;
	
	
		for( int n=0; n < numBoundingBoxes; n++ )
		{

			const int nc = confCPU[n*2+1];
			float* bb = bbCPU + (n * 4);
			
			printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
			
			if( nc != lastClass || n == (numBoundingBoxes - 1) )
			{
					
				if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
				{
					printf("detectnet-console:  failed to draw boxes\n");
					
				}
				
				// color 변수를 float4형으로 선언
				float4 color;
				
				color.x = 255.0f; // r 값에 255 대입
				color.y = 0.0f; // g 값에 255 대입
				color.z = 0.0f; // b 값에 255 대입
				color.w = 255.0f; // a 값에 255 대입
		
				cpuRectOutlines <float4>((float4 *)imgCPU, (float4 *)imgCPU, imgWidth, imgHeight, (float4*)(bbCUDA + (lastStart * 4)), (n - lastStart) + 1, color);
				// 이미지 처리를 담당하는 cpuRectOutlines 함수 호출

				lastClass = nc;
				lastStart = n;
			}
		}
		
		CUDA(cudaThreadSynchronize());
		
		// save image to disk
		printf("detectnet-console:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
		
		if( !saveImageRGBA(argv[2], (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
			printf("detectnet-console:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
		else	
			printf("detectnet-console:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
		
	}
	//printf("detectnet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, "pedestrian");
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;

	// 이렇게 gpioUnexport를 주석 친 이유는 프로그램이 종료된 후에도 led를 켜주기 위함이다
	/*gpioUnexport(led0);	
	gpioUnexport(led1);	
	gpioUnexport(led2);	
	gpioUnexport(led3);*/	

	return 0;
}
