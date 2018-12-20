#pragma once

/**
* Contains image informations
* - raw:	row-major order 2D matrix with pixel's intensities
* - height:	number of columns
* - width:	number of rows
* - bpp:	bits per pixel
* - pitch:	number of bytes of a row rounded to next 32 bit
*/
typedef struct _GrayScaleImage {
	unsigned char* raw;
	unsigned int height;
	unsigned int width;
	unsigned int bpp;
	unsigned int pitch;
} GrayScaleImage;

/**
* Loads an image from filesystem
* args:
*	- const char* path:	absolute path of the image
* return:	a pointer to a grayScaleImage or NULL if file not found or image is not gray scale (8 bits per pixels)
*/
GrayScaleImage* buildGrayScaleImage(const char* path);

/**
* Create a grayScaleImage with width * height pixels of random intensities
* args:
*	const unsigned int width:	number of columns
*	const unsigned int height:	number of rows
* return:	a pointer to a grayScaleImage
*/
GrayScaleImage* createFakeImage(const unsigned int width, const unsigned int height);

/**
* Frees memory allocated for the grayScaleImage
* args:
*	- GrayScaleImage* grayScaleImage:	a pointer to the grayScaleImage that must be freed
*/
void releaseGrayScaleImage(GrayScaleImage* grayScaleImage);