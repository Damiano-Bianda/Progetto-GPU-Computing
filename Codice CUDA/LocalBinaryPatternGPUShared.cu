#pragma once

#include <math.h>
#include <stdio.h>

#include "LocalBinaryPatternGPUShared.h"
#include "Utils.h"
#include "LocalBinaryPattern.h"

// contains:
// - in case of bilinear interpolation: top-left coordinates of pixels involved in bilinear calculation near the points on the circumference
// - in case of nearest neighbor: integer coordinates of nearest pixels to the points on the circumference
__constant__ static Point neighboroodRelativeCoordinates[MAX_POINTS];

// contains bilinear parameters alfa and beta used to calculate bilinear interpolation
__constant__ static PointF bilinearFilterParameters[MAX_POINTS];

/**
* Copy length points to the constant memory 1D array neighboroodRelativeCoordinates
*/
void copyNeighborhoodCoordsInConstantMemory(Point* points, int length) {
	CHECK(cudaMemcpyToSymbol(neighboroodRelativeCoordinates, points, length * sizeof(Point)));
}

/**
* Copy length points to the constant memory 1D array bilinearFilterParameters
*/
void copyBilinearParamsInConstantMemory(PointF * points, int length) {
	CHECK(cudaMemcpyToSymbol(bilinearFilterParameters, points, length * sizeof(PointF)));
}


/**
* Calculate the LBP associated with pixel assigned at current thread, it is used by nearest neighbor LBP
* args:
*  - int borderLength:				max distance of pixels that must be reached outside block size
*  - unsigned char* sharedMemory:	pointer to the shared memory containing a portion of the image
*  - int sharedWidth:				width of the shared memory
*  - int sharedHeight:				height of the shared memory
* return: a 32 bit value containing the calculated LBP
* template <int points>:	at compile time will be generated more compiled version of this function
*/
template <int points>
__device__ static unsigned int calculateLocalBinaryPatternNearestNeighbor(int borderLength, unsigned char* sharedMemory, int sharedWidth, int sharedHeight) {
	// coordinate of central pixel in shared memory
	Point centralPixelCoordinateShared{ threadIdx.x + borderLength, threadIdx.y + borderLength };
	unsigned char centralPixelIntensity = sharedMemory[centralPixelCoordinateShared.y * sharedWidth + centralPixelCoordinateShared.x];
	unsigned int LBPValue = 0;
#pragma unroll points
	for (int p = 0; p < points; p++) {
		unsigned char neighborIntensity = sharedMemory[(centralPixelCoordinateShared.y + neighboroodRelativeCoordinates[p].y) * sharedWidth + (centralPixelCoordinateShared.x + neighboroodRelativeCoordinates[p].x)];
		// write 1 or 0 in last bit of LBP
		LBPValue <<= 1;
		LBPValue |= (neighborIntensity >= centralPixelIntensity);
	}
	return LBPValue;
}

/**
* Calculate the LBP associated with pixel assigned at current thread, it is used by bilinear interpolation LBP
* args:
*  - int borderLength:				max distance of pixels that must be reached outside block size
*  - unsigned char* sharedMemory:	pointer to the shared memory containing a portion of the image
*  - int sharedWidth:				width of the shared memory
*  - int sharedHeight:				height of the shared memory
* return: a 32 bit value containing the calculated LBP
* template <int points>:	at compile time will be generated more compiled version of this function
*/
template <int points>
__device__ static unsigned int calculateLocalBinaryPatternBilinear(int borderLength, unsigned char* sharedMemory, int sharedWidth, int sharedHeight) {
	Point centralPixelCoordinateShared{ threadIdx.x + borderLength, threadIdx.y + borderLength };
	float centralPixelIntensity = sharedMemory[centralPixelCoordinateShared.y * sharedWidth + centralPixelCoordinateShared.x];
	unsigned int LBPValue = 0;

#pragma unroll points
	for (int p = 0; p < points; p++) {
		
		// retrieve top-left, top-right, bottom-left, bottom-right pixels intensities involved in bilinear interpolation
		unsigned char bilinearNeighborhood[BILINEAR_NEIGHBORHOOD];
		unsigned int x = centralPixelCoordinateShared.x + neighboroodRelativeCoordinates[p].x;
		unsigned int y = centralPixelCoordinateShared.y + neighboroodRelativeCoordinates[p].y;
		bilinearNeighborhood[0] = sharedMemory[y * sharedWidth + x];
		bilinearNeighborhood[1] = sharedMemory[y * sharedWidth + (x + 1)];
		bilinearNeighborhood[2] = sharedMemory[(y + 1) * sharedWidth + x];
		bilinearNeighborhood[3] = sharedMemory[(y + 1) * sharedWidth + (x + 1)];

		// calculate interpolated intensity
		float alfa = bilinearFilterParameters[p].x;
		float beta = bilinearFilterParameters[p].y;
		float neighborIntensity =
			bilinearNeighborhood[0] * (1 - alfa) * (1 - beta) +
			bilinearNeighborhood[1] * alfa * (1 - beta) +
			bilinearNeighborhood[2] * (1 - alfa) * beta +
			bilinearNeighborhood[3] * alfa * beta;
		
		// write 1 or 0 in last bit of LBP
		LBPValue <<= 1;
		LBPValue |= (neighborIntensity >= centralPixelIntensity);
	}

	return LBPValue;
}

/**
* Given a circular binary word of a certain length, count the number of switches 10 or 01
* args:
*  unsigned int word:	the binary word where the switches will be counted
*  int length:			the length of the binary word, beginning from least significant bit
*  unsigned int mask:	a mask of length least significant bits set at 1
* return: return the number of switches in the circular binary word
*/
__device__ static unsigned int countBitSwitches(unsigned int word, int length, unsigned int mask) {
	unsigned int rotatedWord = ((word >> 1) | (word << (length - 1))) & mask;
	unsigned int xor = word ^ rotatedWord;
	// __popc counts the number of bits set at 1
	return __popc(xor);
}

/**
* Check uniformity of LBPValue and return appropriate histogram index
* args:
*  - unsigned int LBPValue:				the value of the LBP
*  - unsigned int numberOfBitSwitches:	number of transictions 01 and 10 in the LBP
*  - int points:						number of points used by LBP algorithm
*  - return:	the correct histogram index
*/
__device__ static unsigned int calculateHistogramIndex(unsigned int LBPValue, unsigned int numberOfBitSwitches, int points) {
	return numberOfBitSwitches <= 2 ? __popc(LBPValue) : points + 1;
}

/**
* Increment of 1 the binIndex bin of the histogram, this operation is atomic in order to avoid race conditions
* args:
*  - Histogram* histogram:			pointer to the histogram that must be updated
*  - unsigned int binIndex:			the index of the bin that will be incremented
*/
__device__ static void incrementBin(Histogram histogram, unsigned int binIndex) {
	atomicAdd(&histogram.bins[binIndex], 1);
}

/**
* Each block of threads copy in shared memory the portion of image that will access during LBP algorithm.
* The portion is the submatrix with pixels where single threads of the block must calculate LBP plus a borderLength padding in order to access neighborhood.
* In case of padding outside the borders of the image, shared memory is filled with 0.
* At the end of the function threads will synchronize.
* args:
*  grayScaleImage grayScaleImage:	the struct containing image infos and pointer to raw data
*  unsigned char* sharedTile:		pointer to the first element of shared memory
*  int borderLength:				size of padding
*  int sharedWidth:					width of shared memory
*  int sharedHeight:				height of shared memory
*/
__device__ static void copyImageToSharedMemory(GrayScaleImage grayScaleImage, unsigned char* sharedTile, int borderLength, int sharedWidth, int sharedHeight) {
	// setup initial offsets
	const int startX = -borderLength;
	const int endX = blockDim.x + borderLength;
	const int startY = -borderLength;
	const int endY = blockDim.y + borderLength;

	// global coordinate of the thread
	int globalY = blockIdx.y * blockDim.y + threadIdx.y;
	int globalX = blockIdx.x * blockDim.x + threadIdx.x;
	
	// local (shared) coordinate of the thread
	int sharedY = threadIdx.y;

	// double for iterate over offsets in two dimension in order to update shared and global coordinates
	for (int offsetY = startY; offsetY < endY; offsetY += blockDim.y) {

		int globalMovedY = globalY + offsetY;

		int sharedX = threadIdx.x;
		for (int offsetX = startX; offsetX < endX; offsetX += blockDim.x) {
			int globalMovedX = globalX + offsetX;

			// check if a thread is writing in a valid shared memory position, if not is one thread that has already written all of his values
			if (0 <= sharedX && sharedX < sharedWidth && 0 <= sharedY && sharedY < sharedHeight) {
				// check if a thread is reading from a valid global memory position, if not i write 0 (0-padding) in shared memory
				if (0 <= globalMovedX && globalMovedX < grayScaleImage.width && 0 <= globalMovedY && globalMovedY < grayScaleImage.height) {
					sharedTile[sharedY * sharedWidth + sharedX] = grayScaleImage.raw[globalMovedY * grayScaleImage.width + globalMovedX];
				}
				else {
					sharedTile[sharedY * sharedWidth + sharedX] = 0;
				}
			}
			sharedX += blockDim.x;
		}
		sharedY += blockDim.y;
	}
	__syncthreads();
}

/**
* Main kernel, execute the LBP on image producing the resulting histogram.
* args:
*  GrayScaleImage image:		cointains image informations and preallocated device pointer to raw data 
*  Histogram resultHistogram:	cointains histogram informations and preallocated device pointer to raw data, is the return value of the kernel
*  unsigned int mask:			precalculated bitmask with points least significant bit set at 1
*  int borderLength:			max size outside the block size that a thread on a block border must access
* template<bool bilinear, int points>:	at compile time will be generated more compiled version of this function, in order to unroll loop and others compiler optimizations
*/
template<bool bilinear, int points>
__global__ void localBinaryPatternDeviceShared(GrayScaleImage image, Histogram resultHistogram, unsigned int mask, int borderLength) {

	extern __shared__ unsigned char tile[];

	int sharedHeight = blockDim.y + 2 * borderLength;
	int sharedWidth = blockDim.x + 2 * borderLength;

	copyImageToSharedMemory(image, tile, borderLength, sharedWidth, sharedHeight);


	// each thread corresponding to an image pixel is responsible for its local binary pattern calculation, surplus threads are discarted.
	if (blockDim.x * blockIdx.x + threadIdx.x >= image.width
		|| blockDim.y * blockIdx.y + threadIdx.y >= image.height) {
		return;
	}

	unsigned int LBPValue = bilinear ? 
		calculateLocalBinaryPatternBilinear<points>(borderLength, tile, sharedWidth, sharedHeight) :
		calculateLocalBinaryPatternNearestNeighbor<points>(borderLength, tile, sharedWidth, sharedHeight);
	unsigned int bitSwitches = countBitSwitches(LBPValue, points, mask);
	unsigned int histogramIndex = calculateHistogramIndex(LBPValue, bitSwitches, points);
	incrementBin(resultHistogram, histogramIndex);

}

/**
* Template function that implements a for loop through recursion, in order to create compiled kernels with different params.
* Body of the function is a loop iteration that check if current iteration is equal to run time specified iteration,
* if true launch right kernel, else use recursion to do next iteration
* args:
*  int dynamicDimension:	run-time specified iteration
*  bool bilinear:			launch bilinear or nearest neighbor kernel
*  int sharedSize:			shared memory bytes reserved for each block of threads
*  GrayScaleImage d_image:	cointains image informations and preallocated device pointer to raw data 
*  Histogram d_histogram:   cointains histogram informations and preallocated device pointer to raw data, is the return value of the kernel
*  unsigned int mask:		precalculated bitmask with points least significant bit set at 1
*  int borderLength:		max size outside the block size that a thread on a block border must access
*  template<int staticDimension>:	every iteration is compiled with growing staticDimension [1-32]
*/
template<int staticIteration>
void selectKernel(int dynamicIteration, bool bilinear, int sharedSize, GrayScaleImage d_image, Histogram d_histogram, unsigned int mask, int borderLength) {
	if (dynamicIteration == staticIteration)
	{
		if (bilinear)
			localBinaryPatternDeviceShared <true, staticIteration> << <getGridSize(d_image), BLOCK_SIZE, sharedSize >> >(d_image, d_histogram, mask, borderLength);
		else {
			localBinaryPatternDeviceShared <false, staticIteration> << <getGridSize(d_image), BLOCK_SIZE, sharedSize >> >(d_image, d_histogram, mask, borderLength);
		}
	}
	selectKernel<staticIteration + 1>(dynamicIteration, bilinear, sharedSize, d_image, d_histogram, mask, borderLength);
}

/**
* Final loop iteration, P is comprised between 1 and 32
* args: same of basic selectKernel function
*/
template<>
void selectKernel<MAX_POINTS + 1>(int dynamicIteration, bool bilinear, int sharedSize, GrayScaleImage d_image, Histogram d_histogram, unsigned int mask, int borderLength)
{
	// do nothing, but end recursion
}

/**
* First loop iteration, P start from 1
* args: same of basic selectKernel function
*/
void selectKernel(int dynamicIteration, bool bilinear, int sharedSize, GrayScaleImage d_image, Histogram d_histogram, unsigned int mask, int borderLength)
{
	selectKernel<MIN_POINTS>(dynamicIteration, bilinear, sharedSize, d_image, d_histogram, mask, borderLength);
}

/**
* Utility function that helps to find max coordinate (x or y) in an array of Points
* args:
*  Point* point:	a pointer to the first element of the array
*  int length:		the length of the array
* return:	the max coordinate element present in this array
*/
int findMaxElement(Point* points, int length) {
	int max = points[0].x;
	for (int i = 0; i < length; i++) {
		if (points[i].x > max) {
			max = points[i].x;
		}
		if (points[i].y > max) {
			max = points[i].y;
		}
	}
	return max;
}

Histogram* LocalBinaryPatternGPUSharedMemory(GrayScaleImage* grayScaleImage, int points, float radius, bool bilinear, double* time) {
	
	// prepare kernel input and output data
	unsigned int bitmask = createMask(points);
	GrayScaleImage d_image = *grayScaleImage;
	Histogram* h_histogram = buildHistogram(histogramBinsCount(points));
	Histogram d_histogram = *h_histogram;

	// start cuda timer
	TimerCuda* timer = buildTimerCuda();
	startClock(timer);

	// alloc gpu memory and copy image data and an empty histogram on device
	CHECK(cudaMalloc(&d_image.raw, d_image.width * d_image.height * sizeof(unsigned char)));
	CHECK(cudaMalloc(&d_histogram.bins, d_histogram.binsCount * sizeof(unsigned int)));
	CHECK(cudaMemcpy(d_image.raw, grayScaleImage->raw, d_image.width * d_image.height * sizeof(unsigned char), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_histogram.bins, h_histogram->bins, d_histogram.binsCount * sizeof(unsigned int), cudaMemcpyHostToDevice));

	Point* h_neighborhoodCoordinates;
	if (bilinear) {
		PointF* h_bilinearParameters;
		// given points and radius create two arrays: first contains top-left bilinear coordinates of neighborhood, second parameters alfa and beta (bilinear interpolation)
		surroundingBilinearPoints(points, radius, &h_neighborhoodCoordinates, &h_bilinearParameters);
		// copy to constant memory
		copyNeighborhoodCoordsInConstantMemory(h_neighborhoodCoordinates, points);
		copyBilinearParamsInConstantMemory(h_bilinearParameters, points);

		// compute size of shared memory, based on blocksize and border on which threads must read
		int borderLength = findMaxElement(h_neighborhoodCoordinates, points) + 1;
		int sharedSize = (BLOCK_SIZE.x + 2 * borderLength) * (BLOCK_SIZE.y + 2 * borderLength) * sizeof(unsigned char);
		
		// launch kernel and check for errors
		selectKernel(points, bilinear, sharedSize, d_image, d_histogram, bitmask, borderLength);
		CHECK(cudaGetLastError());

		// free allocated memory
		releasePoint(h_bilinearParameters);
	}
	else {
		// calculate nearest neighbor coordinates of neighborhood, and copy to constant memory
		h_neighborhoodCoordinates = surroundingPoints(points, radius);
		copyNeighborhoodCoordsInConstantMemory(h_neighborhoodCoordinates, points);

		// compute size of shared memory, based on blocksize and border on which threads must read
		int borderLength = findMaxElement(h_neighborhoodCoordinates, points);
		int sharedSize = (BLOCK_SIZE.x + 2 * borderLength) * (BLOCK_SIZE.y + 2 * borderLength) * sizeof(unsigned char);
		
		// launch kernel and check for errors
		selectKernel(points, bilinear, sharedSize, d_image, d_histogram, bitmask, borderLength);
		CHECK(cudaGetLastError());
	}
	// retrieve histogram data
	CHECK(cudaMemcpy(h_histogram->bins, d_histogram.bins, d_histogram.binsCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	// get elapsed time
	endClock(timer);
	synchronize(timer); 
	*time = (double)elapsedMilliSeconds(timer);
	releaseTimer(timer);

	// free memory host and device
	releasePoint(h_neighborhoodCoordinates);
	CHECK(cudaFree(d_image.raw));
	CHECK(cudaFree(d_histogram.bins));

	return h_histogram;
}