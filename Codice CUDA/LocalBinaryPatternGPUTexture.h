#pragma once

#include "GrayScaleImage.h"
#include "Histogram.h"
#include "Utils.h"

/**
* Launch a parallel execution of LBP on GPU using texture memory
* args:
*  GrayScaleImage* grayScaleImage:	a pointer to the image on which the LBP is calculated
*  const int points:				number of points of the circumference
*  const int radius:				radius of the circumference
*  bool bilinear:					set the sampling methods, true for bilinear interpolation or false for nearest neighbor
*  double* time:					after execution this variabile will be written with time of execution
* return:	a pointer to an histogram, that is the result of LBP
*/
Histogram* LocalBinaryPatternGPUTextureMemory(GrayScaleImage* grayScaleImage, int points, float radius, bool bilinear, double* time);