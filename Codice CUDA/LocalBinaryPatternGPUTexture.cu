#pragma once

#include <math.h>
#include <stdio.h>

#include "LocalBinaryPatternGPUTexture.h"
#include "Utils.h"
#include "LocalBinaryPattern.h"


// reference to the texture used during nearest neighbor execution, read an unsigned char from a 2D texture memory
texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> textureReferenceNearestNeighbor;

// reference to the texture used during bilinear interpolation execution, read an unsigned char from a 2D texture memory
texture<unsigned char, cudaTextureType2D, cudaReadModeNormalizedFloat> textureReferenceBilinear;

// contains integer coordinates of nearest pixels to the points on the circumference, used for nearest neighbor
__constant__ static Point neighboroodRelativeCoordinatesNN[MAX_POINTS];

// contains float coordinates of points on the circumference, used for bilinear interpolation
__constant__ static PointF neighboroodRelativeCoordinatesBilinear[MAX_POINTS];

/**
* Copy length points to the constant memory 1D array neighboroodRelativeCoordinatesNN, used for nearest neighbor
*/
void copyNeighborhoodCoordsInConstantMemoryNN(Point* points, int length) {
	CHECK(cudaMemcpyToSymbol(neighboroodRelativeCoordinatesNN, points, length * sizeof(Point)));
}

/**
* Copy length points to the constant memory 1D array neighboroodRelativeCoordinatesBilinear, used for bilinear interpolation
*/
void copyNeighborhoodCoordsInConstantMemoryBilinear(PointF* points, int length) {
	CHECK(cudaMemcpyToSymbol(neighboroodRelativeCoordinatesBilinear, points, length * sizeof(PointF)));
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
* Calculate the LBP associated with pixel assigned at current thread, it is used by nearest neighbor LBP
* args:
*  - Point centralPixelCoordinate:	the coordinate on which LBP is calculated
* return: a 32 bit value containing the calculated LBP
* template <int points>:	at compile time will be generated more compiled version of this function
*/
template <int points>
__device__ static unsigned int calculateLocalBinaryPatternNearestNeighbor(Point centralPixelCoordinate) {
	PointF textureCoordinate = { centralPixelCoordinate.x + 0.5f, centralPixelCoordinate.y + 0.5f };
	unsigned int LBPValue = 0;
	unsigned char centralPixelIntensity = tex2D(textureReferenceNearestNeighbor, textureCoordinate.x, textureCoordinate.y);

#pragma unroll points
	for (int p = 0; p < points; p++) {
		unsigned char neighborIntensity = tex2D(textureReferenceNearestNeighbor, textureCoordinate.x + neighboroodRelativeCoordinatesNN[p].x, textureCoordinate.y + neighboroodRelativeCoordinatesNN[p].y);		
		// write 1 or 0 in last bit of LBP
		LBPValue <<= 1;
		LBPValue |= (neighborIntensity >= centralPixelIntensity);
	}
	return LBPValue;
}

/**
* Calculate the LBP associated with pixel assigned at current thread, it is used by bilinear interpolation LBP
* args:
*  - Point centralPixelCoordinate:	the coordinate on which LBP is calculated
* return: a 32 bit value containing the calculated LBP
* template <int points>:	at compile time will be generated more compiled version of this function
*/
template <int points>
__device__ static unsigned int calculateLocalBinaryPatternBilinear(Point centralPixelCoordinate) {
	PointF textureCoords = { centralPixelCoordinate.x + 0.5f, centralPixelCoordinate.y + 0.5f };
	unsigned int LBPValue = 0;
	float centralPixelIntensity = tex2D(textureReferenceBilinear, textureCoords.x, textureCoords.y);

#pragma unroll points
	for (int p = 0; p < points; p++) {
		float neighborIntensity = tex2D(textureReferenceBilinear, textureCoords.x + neighboroodRelativeCoordinatesBilinear[p].x , textureCoords.y + neighboroodRelativeCoordinatesBilinear[p].y);
		// write 1 or 0 in last bit of LBP
		LBPValue <<= 1;
		LBPValue |= (neighborIntensity >= centralPixelIntensity);

	}
	return LBPValue;
}

/**
* Main kernel, execute the LBP on image producing the resulting histogram.
* args:
*  Histogram resultHistogram:	cointains histogram informations and preallocated device pointer to raw data, is the return value of the kernel
*  unsigned int mask:			precalculated bitmask with points least significant bit set at 1
*  int width:					width of the image
*  int height:					height of the image
* template<bool bilinear, int points>:	at compile time will be generated more compiled version of this function, in order to unroll loop and others compiler optimizations
*/
template <bool bilinear, int points>
__global__ void localBinaryPatternDeviceTexture(Histogram resultHistogram, unsigned int mask, int width, int height)
{
	Point centralPixelCoordinate{ blockDim.x * blockIdx.x + threadIdx.x, blockDim.y * blockIdx.y + threadIdx.y };
	
	// each thread corresponding to an image pixel is responsible for its local binary pattern calculation
	// other threads are discarted.
	if (centralPixelCoordinate.x >= width || centralPixelCoordinate.y >= height) {
		return;
	}

	unsigned int LBPValue = bilinear ? calculateLocalBinaryPatternBilinear<points>(centralPixelCoordinate) : calculateLocalBinaryPatternNearestNeighbor<points>(centralPixelCoordinate);
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
*  Histogram d_histogram:   cointains histogram informations and preallocated device pointer to raw data, is the return value of the kernel
*  unsigned int mask:		precalculated bitmask with points least significant bit set at 1
*  GrayScaleImage d_image:	cointains image informations and preallocated device pointer to raw data
*  template<int staticDimension>:	every iteration is compiled with growing staticDimension [1-32]
*/
template<int staticIteration>
void selectKernel(int dynamicIteration, bool bilinear, Histogram d_histogram, unsigned int mask, GrayScaleImage* h_image) {
	if (dynamicIteration == staticIteration)
	{
		if (bilinear)
			localBinaryPatternDeviceTexture <true, staticIteration> << <getGridSize(*h_image), BLOCK_SIZE >> >(d_histogram, mask, h_image->width, h_image->height);
		else {
			localBinaryPatternDeviceTexture <false, staticIteration> << <getGridSize(*h_image), BLOCK_SIZE >> >(d_histogram, mask, h_image->width, h_image->height);
		}
	}
	selectKernel<staticIteration + 1>(dynamicIteration, bilinear, d_histogram, mask, h_image);
}

/**
* Final loop iteration, P is comprised between 1 and 32
* args: same of basic selectKernel function
*/
template<>
void selectKernel<MAX_POINTS + 1>(int dynamicIteration, bool bilinear, Histogram d_histogram, unsigned int mask, GrayScaleImage* h_image)
{
	// do nothing, but end recursion
}

/**
* First loop iteration, P start from 1
* args: same of basic selectKernel function
*/

void selectKernel(int dynamicIteration, bool bilinear, Histogram d_histogram, unsigned int mask, GrayScaleImage* h_image)
{
	selectKernel<MIN_POINTS>(dynamicIteration, bilinear, d_histogram, mask, h_image);
}

Histogram* LocalBinaryPatternGPUTextureMemory(GrayScaleImage* h_image, int points, float radius, bool bilinear, double* time) {

	// prepare kernel input and output data
	unsigned int mask = createMask(points);
	Histogram* h_histogram = buildHistogram(histogramBinsCount(points));
	Histogram d_histogram = *h_histogram;

	// start cuda timer
	TimerCuda* timer = buildTimerCuda();
	startClock(timer);

	// alloc gpu memory and copy image data and an empty histogram on device
	CHECK(cudaMalloc(&d_histogram.bins, d_histogram.binsCount * sizeof(unsigned int)));
	CHECK(cudaMemcpy(d_histogram.bins, h_histogram->bins, d_histogram.binsCount * sizeof(unsigned int), cudaMemcpyHostToDevice));
	unsigned char* d_image;
	size_t pitch;
	CHECK(cudaMallocPitch(&d_image, &pitch, h_image->width * sizeof(unsigned char), h_image->height));
	CHECK(cudaMemcpy2D(d_image, pitch, h_image->raw,
		h_image->width * sizeof(unsigned char), h_image->width * sizeof(unsigned char),
		h_image->height, cudaMemcpyHostToDevice));

	if (bilinear) {

		// Set texture reference parameters: cudaAddressModeBorder is 0-padding and cudaFilterModeLinear is bilinear interpolation
		textureReferenceBilinear.addressMode[0] = cudaAddressModeBorder;
		textureReferenceBilinear.addressMode[1] = cudaAddressModeBorder;
		textureReferenceBilinear.filterMode = cudaFilterModeLinear;

		// bind texture reference to array that contains image data
		size_t bytesOffset;
		CHECK(cudaBindTexture2D(&bytesOffset, &textureReferenceBilinear, d_image, &textureReferenceBilinear.channelDesc, h_image->width, h_image->height, pitch));
		
		// retrieve float coordinates of points over the circumference and load them in constant memory
		PointF* h_neighborhoodCoordinates = surroundingPointsF(points, radius);
		copyNeighborhoodCoordsInConstantMemoryBilinear(h_neighborhoodCoordinates, points);

		// launch kernel and check for error
		selectKernel(points, bilinear, d_histogram, mask, h_image);
		CHECK(cudaGetLastError());

		// free allocated memory
		releasePoint(h_neighborhoodCoordinates);
	}
	else {
		
		// Set texture reference parameters: cudaAddressModeBorder is 0-padding and cudaFilterModePoint is nearest neighbor
		textureReferenceNearestNeighbor.addressMode[0] = cudaAddressModeBorder;
		textureReferenceNearestNeighbor.addressMode[1] = cudaAddressModeBorder;
		textureReferenceNearestNeighbor.filterMode = cudaFilterModePoint;
		
		// bind texture reference to array that contains image data
		size_t bytesOffset;
		CHECK(cudaBindTexture2D(&bytesOffset, &textureReferenceNearestNeighbor, d_image, &textureReferenceNearestNeighbor.channelDesc, h_image->width, h_image->height, pitch));
		
		// retrieve int coordinates of points near the circumference and load them in constant memor
		Point* h_neighborhoodCoordinates = surroundingPoints(points, radius);
		copyNeighborhoodCoordsInConstantMemoryNN(h_neighborhoodCoordinates, points);

		// launch kernel and check for error
		selectKernel(points, bilinear, d_histogram, mask, h_image);
		CHECK(cudaGetLastError());

		// free allocated memory
		releasePoint(h_neighborhoodCoordinates);
	}
	// retrieve histogram result
	CHECK(cudaMemcpy(h_histogram->bins, d_histogram.bins, d_histogram.binsCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// compute elapsed time
	endClock(timer);
	synchronize(timer);
	*time = (double)elapsedMilliSeconds(timer);

	// free host and device memory
	releaseTimer(timer);
	CHECK(cudaFree(d_image));
	CHECK(cudaFree(d_histogram.bins));

	return h_histogram;
}