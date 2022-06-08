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

// �̹����� ó�����ִ� cpuRectOutlines �Լ� ����
template <typename T> // template �� �̿� 
void cpuRectOutlines(T* input, T* output, int width, int height, float4* rects, int numRects, float4 color ) 
{	

	T* tmp = (T*)malloc(sizeof(T)*width*height); 
	// tmp�� ���� �̹����� ��� �ȼ��� �ǹ��ϴ� input�� �ӽú�����, �̸� �����Ҵ��Ͽ� �����
	
	// height * width ��ŭ ���� �ݺ����� ���鼭
	for(int i = 0 ; i < height ; i ++){ 
		for(int j = 0 ; j < width ; j++){
			tmp[i*width + j] = input[i*width + j];	// input �迭�� ���纻�� tmp�� input�� i,j�� �ش��ϴ� ���� ��� ����
		}
	}

	// �̹��� �߰��� �̹��� ���� ũ���� �簢�� �׵θ��� �׸�
	for(int i = (int)((height)/4.) ; i <= (int)((height*3)/4.); i ++){
		for(int j = (int)((width)/4.); j <= (int)((width*3)/4.); j ++){
		/* �̹��� �߰��� �̹��� ���� ũ���� �簢�� �׵θ��� �׸��� ���ؼ��� 
		���� �ٱ��� �ݺ��������� height*1/4~height*3/4 ������ ���� �ݺ��ϸ鼭
		���� �ݺ��������� width*1/4~width*3/4 ������ ���� �ݺ��ϸ鼭
		
		(�� ���̰��� ���� ������ �̹��� ���� ũ���� �簢�� ���θ� Ȯ���ϸ鼭 �׵θ��� �ش��ϴ� ���� �ִ��� Ȯ���ϱ� �����̴�) 
		*/
			if((i == (int)((height)/4.)) || (i == (int)((height*3)/4.))){ // ���� i�� �׵θ�(height/4, height*3/4)�� �ش��Ѵٸ�
				// ���������� ĥ��
				tmp[i*width + j].x = color.x;
				tmp[i*width + j].y = color.y;
				tmp[i*width + j].z = color.z;
				tmp[i*width + j].w = color.w;
			}

			if((j == (int)((width)/4.)) || (j == (int)((width*3)/4.))){ // ���� j�� �׵θ�(width/4, width*3/4)�� �ش��Ѵٸ�
				// ���������� ĥ��
				tmp[i*width + j].x = color.x;
				tmp[i*width + j].y = color.y;
				tmp[i*width + j].z = color.z;
				tmp[i*width + j].w = color.w;
			}
			
		}
	}



	int count = 0; // �׸� �߾��� �簢�� �׵θ��� ���� ���� ������ �����ϴ� ����

	for( int nr = 0; nr < numRects; nr++ ) // �簢���� ������ŭ �ݺ����� ���鼭
	{
		const float4 r = rects[nr]; // r�̶�� ������ rects[nr]�� ���� ����
		//printf("=>%f %f %f %f", r.x,r.y,r.z,r.w);
	
		float xid = (r.x + r.z) / 2; // xid �� �̹����� ���� �߰������� ����
		float yid = (r.y + r.w) / 2; // yid �� �̹����� ���� �߰������� ����
		
		/*�Ʒ����� ������ ����� ������ bounding box�� ��� ���� ���� ��, �� �ȼ��� ������ ���� ������ �����Ƿ�,
		��� ���� �ش��ϴ� �ȼ��� �ֺ��� �� �� 9���� ���� ��� ���� ���� �� �ֵ��� �ϴ� ���̴�.
		���� bounding box�� ��� ���� tmp[yid][xid]�� �ش��ϴ� ���̰�, �ֺ��� �ȼ����� ���������� ��ĥ�� ���� �� ���� �� �ֵ��� �ϱ� �����̴�*/
		
		// tmp[yid-1][xid-1]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid-1 + ((int)yid-1) * width].x = color.x; 
		tmp[(int)xid-1 + ((int)yid-1) * width].y = color.y;
		tmp[(int)xid-1 + ((int)yid-1) * width].z = color.z;
		tmp[(int)xid-1 + ((int)yid-1) * width].w = color.w;	
		
				
		// tmp[yid][xid-1]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid-1 + ((int)yid) * width].x = color.x;
		tmp[(int)xid-1 + ((int)yid) * width].y = color.y;
		tmp[(int)xid-1 + ((int)yid) * width].z = color.z;
		tmp[(int)xid-1 + ((int)yid) * width].w = color.w;

		// tmp[yid+1][xid-1]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid-1 + ((int)yid+1) * width].x = color.x;
		tmp[(int)xid-1 + ((int)yid+1) * width].y = color.y;
		tmp[(int)xid-1 + ((int)yid+1) * width].z = color.z;
		tmp[(int)xid-1 + ((int)yid+1) * width].w = color.w;	

  		// tmp[yid-1][xid]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid + ((int)yid-1) * width].x = color.x;
		tmp[(int)xid + ((int)yid-1) * width].y = color.y;
		tmp[(int)xid + ((int)yid-1) * width].z = color.z;
		tmp[(int)xid + ((int)yid-1) * width].w = color.w;	
	
		 
		// tmp[yid][xid]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid + (int)yid * width].x = color.x;
		tmp[(int)xid + (int)yid * width].y = color.y;
		tmp[(int)xid + (int)yid * width].z = color.z;
		tmp[(int)xid + (int)yid * width].w = color.w;

		// tmp[yid+1][xid]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid  + ((int)yid+1) * width].x = color.x;
		tmp[(int)xid  + ((int)yid+1) * width].y = color.y;
		tmp[(int)xid  + ((int)yid+1) * width].z = color.z;
		tmp[(int)xid  + ((int)yid+1) * width].w = color.w;	

		// tmp[yid-1][xid+1]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		input[(int)xid+1  + ((int)yid-1) * width].x = color.x;
		input[(int)xid+1  + ((int)yid-1) * width].y = color.y;
		input[(int)xid+1  + ((int)yid-1) * width].z = color.z;
		input[(int)xid+1  + ((int)yid-1) * width].w = color.w;

		// tmp[yid][xid+1]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid+1 + (int)yid * width].x = color.x;
		tmp[(int)xid+1 + (int)yid * width].y = color.y;
		tmp[(int)xid+1 + (int)yid * width].z = color.z;
		tmp[(int)xid+1 + (int)yid * width].w = color.w;

 		
		// tmp[yid+1][xid+1]�� �ش��ϴ� �ȼ��� ���������� ĥ��
		tmp[(int)xid+1 + ((int)yid+1) * width].x = color.x;
		tmp[(int)xid+1 + ((int)yid+1) * width].y = color.y;
		tmp[(int)xid+1 + ((int)yid+1) * width].z = color.z;
		tmp[(int)xid+1 + ((int)yid+1) * width].w = color.w;

		
		/*���� �׸� ���� ũ�� �簢�� ���ο� xid, yid �� ���ԵǾ��ִٸ�
		
		���⼭ xid�� �簢�� ���ο� ���ԵǷ��� width/4~width*3/4 ����
		yid�� height/4~height*3/4 ���� ���̿��� ��*/

		if( ((int)xid > (int)((width)/4.)) && ((int)xid < (int)((width)*3/4.)) ){
			if ( ((int)yid > (int)((height)/4.)) && ((int)yid < (int)((height)*3/4.)) )
				count ++; // �׸� �߾��� �簢�� �׵θ��� ���� ���� ������ �����ϴ� ������ count�� ����
		}

	}

	printf("\n\ncount : %d\n", count); // count�� ���

	jetsonTX1GPIONumber led[4] = {gpio37,gpio219,gpio36,gpio63}; // �� led�� �ش��ϴ� gpio���� ����
	
	// �������� �ֽ� �ڵ�
	gpioExport(led[0]); 
	gpioExport(led[1]);	
	gpioExport(led[2]);	
	gpioExport(led[3]);	

	gpioSetDirection(led[0], outputPin);
	gpioSetDirection(led[1], outputPin);
	gpioSetDirection(led[2], outputPin);
	gpioSetDirection(led[3], outputPin);

	for(int i = 0;i < count;i++) { 
		// �߾��� �簢���� ���� �������� �� ������ �����ϴ� ����(count) ��ŭ ���鼭 led�� ��� 
		gpioSetValue(led[i], on);
	}
	
	// height * width ��ŭ ���� �ݺ����� ���鼭
	for(int i = 0 ; i < height ; i ++){
		for(int j = 0 ; j < width ; j++){
			input[i*width + j] = tmp[i*width + j]; 
			// �̹��� ó������ ���Ǿ��� �ӽú��� tmp�� �ٽ� input�� ����
			// �̹��� ó�� �� ����� ���� input�� �����Ű�� ����
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
				
				// color ������ float4������ ����
				float4 color;
				
				color.x = 255.0f; // r ���� 255 ����
				color.y = 0.0f; // g ���� 255 ����
				color.z = 0.0f; // b ���� 255 ����
				color.w = 255.0f; // a ���� 255 ����
		
				cpuRectOutlines <float4>((float4 *)imgCPU, (float4 *)imgCPU, imgWidth, imgHeight, (float4*)(bbCUDA + (lastStart * 4)), (n - lastStart) + 1, color);
				// �̹��� ó���� ����ϴ� cpuRectOutlines �Լ� ȣ��

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

	// �̷��� gpioUnexport�� �ּ� ģ ������ ���α׷��� ����� �Ŀ��� led�� ���ֱ� �����̴�
	/*gpioUnexport(led0);	
	gpioUnexport(led1);	
	gpioUnexport(led2);	
	gpioUnexport(led3);*/	

	return 0;
}
