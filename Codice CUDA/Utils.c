#include "Utils.h"

dim3 getGridSize(GrayScaleImage grayScaleImage) {
	return dim3(
		(grayScaleImage.width + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
		(grayScaleImage.height + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y);
}

int histogramBinsCount(int points) {
	return points + 2;
}

void exitAfterError(string message) {
	cerr << message << endl;
	exit(1);
}

unsigned int createMask(const int numberOfBitsSet) {
	unsigned int mask = 0;
	for (int i = 0; i < numberOfBitsSet; i++) {
		mask <<= 1;
		mask |= 1;
	}
	return mask;
}