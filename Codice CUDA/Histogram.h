#pragma once

/**
* Contains histogram informations
* - bins:		1D array with bin's frequencies
* - binsCount:	number of bins
*/
typedef struct _Histogram{
	unsigned int* bins;
	unsigned int binsCount;
} Histogram;

/**
* Create an histogram with binsCount bins at 0
* args:
*	const unsigned int binsCount:	number of bins
* return:	a pointer to an histogram
*/
Histogram* buildHistogram(const unsigned int binsCount);

/**
* Frees memory allocated for the histogram
* args:
*	- Histogram* histogram:	a pointer to the histogram that must be freed
*/
void releaseHistogram(Histogram* histogram);

/**
* Increment of 1 the binIndex bin of the histogram
* args:
*  - Histogram* histogram:			pointer to the histogram that must be updated
*  - const unsigned int binIndex:	the index of the bin that will be incremented
* return:	0 if binIndex refers to a valid bin, 1 otherwise
*/
int incrementBin(Histogram* histogram, const unsigned int binIndex);