#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Point.h"
#include "GrayScaleImage.h"

using namespace std;

/**
* Checks if CUDA error code is effectively an error
* args:
*	- the value of cudaError_t
*/
#define CHECK(call) { \
	const cudaError_t error = call; \
	if (error != cudaSuccess) { \
		printf("Error: %d:%d, ", __FILE__, __LINE__); \
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1); \
	} \
}

// variable needed to check user input
const string SHARED_DEVICE = "shared";
const string CPU_DEVICE = "cpu";
const string TEXTURE_DEVICE = "texture";

// user input constraints
const int MIN_POINTS = 1;
const int MAX_POINTS = 32;
const float MIN_RADIUS = 1.0f;
const float MAX_RADIUS = 5.0f;

// number of pixels involved in bilinear interpolation calculation
const int BILINEAR_NEIGHBORHOOD = 4;

// dim of block
const dim3 BLOCK_SIZE(16, 16);

/**
* calculate grid size at runtime based on BLOCK_SIZE
* args:
*  GrayScaleImage grayScaleImage:	calculate grid size with image width and height
* return: a dim3 value containing the 2D dimension of grid
*/
dim3 getGridSize(GrayScaleImage grayScaleImage);

/**
* given an LBP points args, calculate the length of the histogram
* args:
*  int points:	the number of points
* return:		the number of bin of the histogram
*/
int histogramBinsCount(int points);

/**
* print an error message and exit from program
* args:
*  string message:	error message
*/
void exitAfterError(string message);

/**
* Generate a bitmask of 32 bit where numberOfBitsSet of least significant bits is set and the others are zero.
* args:
*  const int numberOfBitsSet:	the number of least significant bits set to one
* return:	the bitmask
*/
unsigned int createMask(const int numberOfBitsSet);