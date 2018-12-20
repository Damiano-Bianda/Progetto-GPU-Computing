#pragma once

/**
* This test loads predefined images, calculate LBP with different params and compare new histogram with correct past histogram.
* This test is useful when source code is modified to avoid possible new bugs.
* Test CPU, shared kernel and texture kernel
* args:
*	const char* folderPath:	a path to folder in format (//path//to//folder//),
*							the folder must contains predefined images (480x300.jpg, 960x600.jpg, 1920x1200.jpg, 3840x2400.jpg, 7680x4800.jpg).
*/
void testCompareWithOldHistograms(const char* folderPath);

/**
* This test generates random images, calculate LBP with different params and check that cpu resut is equal to shared kernel result.
* Check if histogram counts are correct (== image.width * image.height) for cpu, shared kernel and texture kernel.
*/
void testLBPOnRandomImages();

