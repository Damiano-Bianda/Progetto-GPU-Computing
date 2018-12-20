#pragma once

#include "GrayScaleImage.h"
#include "Histogram.h"
#include "Utils.h"
#include "Timer.h"

/**
* Launch a serial execution of LBP on CPU using nearest neighbour as sampling method
* args:
*  GrayScaleImage* grayScaleImage:	a pointer to the image on which the LBP is calculated
*  const int points:				number of points over the circumference
*  const int radius:				radius of the circumference
*  double* time:					after execution this variabile will be written with time of execution
* return:	a pointer to an histogram, that is the result of LBP
*/
Histogram* localBinaryPatternCPUNearestNeighbor(GrayScaleImage* grayScaleImage, const int points, const float radius, double* time);

/**
* Launch a serial execution of LBP on CPU using bilinear interpolation as sampling method
* args:
*  GrayScaleImage* grayScaleImage:	a pointer to the image on which the LBP is calculated
*  const int points:				number of points over the circumference
*  const int radius:				radius of the circumference
*  double* time:					after execution this variabile will be written with time of execution
* return:	a pointer to an histogram, that is the result of LBP
*/
Histogram* localBinaryPatternCPUBilinear(GrayScaleImage* grayScaleImage, const int P, const float R, double* time);