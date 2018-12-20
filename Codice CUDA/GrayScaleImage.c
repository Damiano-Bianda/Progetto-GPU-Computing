#include "GrayScaleImage.h"

#include <FreeImage.h>
#include <stdlib.h>
#include <time.h>

GrayScaleImage* buildGrayScaleImage(const char* path) {
	FREE_IMAGE_FORMAT format = FreeImage_GetFileType(path);
	if (format == FIF_UNKNOWN) {
		format = FreeImage_GetFIFFromFilename(path);
	}
	if (format == FIF_UNKNOWN) {
		return NULL;
	}
	FIBITMAP* bitmap = FreeImage_Load(format, path);
	if (bitmap == NULL) {
		return NULL;
	}

	if (FreeImage_GetBPP(bitmap) != 8) {
		return NULL;
	}

	GrayScaleImage* result = (GrayScaleImage*)malloc(sizeof(GrayScaleImage));
	result->height = FreeImage_GetHeight(bitmap);
	result->width = FreeImage_GetWidth(bitmap);
	result->pitch = FreeImage_GetPitch(bitmap);
	result->bpp = FreeImage_GetBPP(bitmap);
	result->raw = (unsigned char*)malloc(result->height * result-> width * sizeof(unsigned char));
	FreeImage_ConvertToRawBits(result -> raw, bitmap, result->pitch, result->bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, TRUE);

	FreeImage_Unload(bitmap);
	return result;
}

GrayScaleImage* createFakeImage(const unsigned int width, const unsigned int height) {
	GrayScaleImage* result = (GrayScaleImage*)malloc(sizeof(GrayScaleImage));
	result->height = height;
	result->width = width;
	result->raw = (unsigned char*)malloc(result->height * result->width * sizeof(unsigned char));

	srand(time(NULL));
	for (int y = 0; y < result->height; y++) {
		for (int x = 0; x < result->width; x++) {
			result->raw[y * result->width + x] = rand();
		}
	}

	return result;
}

void releaseGrayScaleImage(GrayScaleImage* grayScaleImage) {
	free(grayScaleImage->raw);
	free(grayScaleImage);
}