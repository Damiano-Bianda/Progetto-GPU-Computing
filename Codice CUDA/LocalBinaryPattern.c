
#include "LocalBinaryPattern.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "Point.h"

/**
* Given a binary word of a certain length, return the number of bits set at 1
* args:
*  unsigned int word:	the binary word where the bits will be counted
*  int length:			the length of the binary word, beginning from least significant bit
* return: the number of bits set at 1
*/
unsigned int countBitsSet(unsigned int word, int length) {
	unsigned int count = 0;
	for (int i = 0; i < length; i++) {
		count += (word >> i) & 0x1;
	}
	return count;
}

/**
* Given a circular binary word of a certain length, count the number of switches 10 or 01
* args:
*  unsigned int word:	the binary word where the switches will be counted
*  int length:			the length of the binary word, beginning from least significant bit
*  unsigned int mask:	a mask of length least significant bits set at 1
* return: return the number of switches in the circular binary word
*/
unsigned int countBitSwitches(unsigned int word, int length, unsigned int mask) {
	unsigned int rotatedWord = ((word >> 1) | (word << (length - 1))) & mask;
	unsigned int xor = word ^ rotatedWord;
	return countBitsSet(xor, length);
}

/**
* Retrieve a pixel's intensity that is located at coordinate neighborRelativeCoordinate + pixelCoordinate in the image
* args:
*  const GrayScaleImage* grayScaleImage:	a pointer to the image
*  const Point centralPixelCoordinate:		coordinate of pixel
*  const Point neighborRelativeCoordinate:	relative coordinate of neighbor pixel
* return: intensity of the pixel, or zero if new coordinate is out of the bounds of the image
*/
unsigned char getneighborValue(const GrayScaleImage* grayScaleImage, const Point centralPixelCoordinate, const Point neighborRelativeCoordinate) {
	int newX = centralPixelCoordinate.x + neighborRelativeCoordinate.x;
	int newY = centralPixelCoordinate.y + neighborRelativeCoordinate.y;
	int width = grayScaleImage->width;
	int height = grayScaleImage->height;
	if (0 <= newX && newX < width && 0 <= newY && newY < height) {
		return grayScaleImage->raw[newY * width + newX];
	}
	else {
		return 0;
	}
}

/**
* Calculate the LBP associated with pixel at position centralPixelCoordinate, it is used by nearest neighbor LBP
* args:
*  - GrayScaleImage* grayScaleImage:			a pointer to the image
*  - Point centralPixelCoordinate:				a point containing coordinate of central pixel
*  - Point* neighborhoodRelativeCoordinates:	a 1D array of Points containing the relatives coordinates near the circumference
*  - int points:								length of neighborRelativeCoordinates array
* return: a 32 bit value containing the calculated LBP
*/
static unsigned int calculateLocalBinaryPatternWithNearestneighbor(GrayScaleImage* grayScaleImage, Point centralPixelCoordinate, Point* neighboroodRelativeCoordinates, int points) {
	unsigned char centralPixelIntensity = grayScaleImage->raw[centralPixelCoordinate.y * grayScaleImage->width + centralPixelCoordinate.x];
	unsigned int LBPValue = 0;
	for (int p = 0; p < points; p++) {
		unsigned char neighborIntensity = getneighborValue(grayScaleImage, centralPixelCoordinate, neighboroodRelativeCoordinates[p]);
		
		// write 1 or 0 in last bit of LBP
		LBPValue <<= 1;
		LBPValue |= (neighborIntensity >= centralPixelIntensity);
	}
	return LBPValue;
}

/**
* Calculate the LBP associated with pixel at position centralPixelCoordinate, it is used by bilinear interpolation LBP
* args:
*  - GrayScaleImage* grayScaleImage:			a pointer to the image
*  - Point centralPixelCoordinate:				a point containing coordinate of central pixel
*  - Point* neighborhoodRelativeCoordinates:	a 1D array of Points containing the relatives coordinates over the circumference
*  - PointF* bilinearFilterParameters:			a 1D array of PointFs containing the parameters alfa and beta involved in bilinear calculation
*  - int points:								length of neighborRelativeCoordinates array
* return: a 32 bit value containing the calculated LBP
*/
static unsigned int calculateLocalBinaryPatternBilinear(GrayScaleImage* grayScaleImage, Point centralPixelCoordinate, Point* neighboroodRelativeCoordinates, PointF* bilinearFilterParameters, int points) {
	float centralPixelIntensity = grayScaleImage->raw[centralPixelCoordinate.y * grayScaleImage->width + centralPixelCoordinate.x];
	int width = grayScaleImage->width;
	int height = grayScaleImage->height;
	unsigned int LBPValue = 0;
	for (int p = 0; p < points; p++) {
		// iterate over top-left, top-right, bottom-left, bottom-right pixels involved in bilinear calculation
		// in order to get their intensities
		unsigned char bilinearNeighborhoodIntensities[BILINEAR_NEIGHBORHOOD];
		for (int i = 0; i < 2; i++) {
			int newY = centralPixelCoordinate.y + neighboroodRelativeCoordinates[p].y + i;
			for (int j = 0; j < 2; j++) {
				int newX = centralPixelCoordinate.x + neighboroodRelativeCoordinates[p].x + j;
				bilinearNeighborhoodIntensities[i * 2 + j] =
					0 <= newX && newX < width && 0 <= newY && newY < height  ? grayScaleImage->raw[newY * width + newX] : 0;
			}
		}
		// calculate interpolated intensity
		float alfa = bilinearFilterParameters[p].x;
		float beta = bilinearFilterParameters[p].y;
		float neighborIntensity =
			bilinearNeighborhoodIntensities[0] * (1 - alfa) * (1 - beta) +
			bilinearNeighborhoodIntensities[1] * alfa * (1 - beta) +
			bilinearNeighborhoodIntensities[2] * (1 - alfa) * beta +
			bilinearNeighborhoodIntensities[3] * alfa * beta;
		// write 1 or 0 in last bit of LBP
		LBPValue <<= 1;
		LBPValue |= (neighborIntensity >= centralPixelIntensity);
	}
	return LBPValue;
}

/**
* Check uniformity of LBPValue and return appropriate histogram index
* args:
*  - unsigned int LBPValue:				the value of the LBP
*  - unsigned int numberOfBitSwitches:	number of transictions 01 and 10 in the LBP
*  - int points:						number of points used by LBP algorithm
*  - return:	the correct histogram index
*/
unsigned int calculateHistogramIndex(unsigned int LBPValue, unsigned int numberOfBitSwitches, int points) {
	return numberOfBitSwitches <= 2 ? countBitsSet(LBPValue, points) : points + 1;
}

Histogram* localBinaryPatternCPUNearestNeighbor(GrayScaleImage* grayScaleImage, const int points, const float radius, double* time) {
	unsigned int mask = createMask(points);
	Point* neighborhoodCoordinates = surroundingPoints(points, radius);
	Histogram* histogram = buildHistogram(points + 2);

	Timer* timer = buildTimer();
	startClock(timer);
	
	for (int y = 0; y < grayScaleImage->height; y++) {
		for (int x = 0; x < grayScaleImage->width; x++) {
			Point centralPixelCoordinate{ x,y };
			unsigned int LBPValue = calculateLocalBinaryPatternWithNearestneighbor(grayScaleImage, centralPixelCoordinate, neighborhoodCoordinates, points);
			unsigned int bitSwitches = countBitSwitches(LBPValue, points, mask);
			unsigned int histogramIndex = calculateHistogramIndex(LBPValue, bitSwitches, points);
			incrementBin(histogram, histogramIndex);
		}
	}

	endClock(timer);
	*time = elapsedMilliSeconds(timer);

	releasePoint(neighborhoodCoordinates);
	releaseTimer(timer);
	return histogram;
}

Histogram* localBinaryPatternCPUBilinear(GrayScaleImage* grayScaleImage, const int points, const float radius, double* time) {
	unsigned int mask = createMask(points);
	Point* neighborhoodCoordinates;
	PointF* bilinearParameters;
	surroundingBilinearPoints(points, radius, &neighborhoodCoordinates, &bilinearParameters);
	Histogram* histogram = buildHistogram(points + 2);

	Timer* timer = buildTimer();
	startClock(timer);

	for (int y = 0; y < grayScaleImage->height; y++) {
		for (int x = 0; x < grayScaleImage->width; x++) {
			Point centralPixelCoordinate{ x,y };
			unsigned int LBPValue = calculateLocalBinaryPatternBilinear(grayScaleImage, centralPixelCoordinate, neighborhoodCoordinates, bilinearParameters, points);
			unsigned int bitSwitches = countBitSwitches(LBPValue, points, mask);
			unsigned int histogramIndex = calculateHistogramIndex(LBPValue, bitSwitches, points);
			incrementBin(histogram, histogramIndex);
		}
	}


	endClock(timer);
	*time = elapsedMilliSeconds(timer);

	releasePoint(neighborhoodCoordinates);
	releasePoint(bilinearParameters);
	releaseTimer(timer);
	return histogram;
}