#include "Histogram.h"

#include <stdio.h>
#include <stdlib.h>

Histogram* buildHistogram(const unsigned int binsCount) {
	Histogram* histogram = (Histogram*)malloc(sizeof(Histogram));
	histogram->bins = (unsigned int*)calloc(binsCount, sizeof(binsCount));
	histogram->binsCount = binsCount;
	return histogram;
}

void releaseHistogram(Histogram* histogram) {
	free(histogram->bins);
	free(histogram);
}

int incrementBin(Histogram* histogram, const unsigned int binIndex) {
	if(binIndex < histogram->binsCount){
		histogram->bins[binIndex] += 1;
		return 0;
	}
	return 1;
}

