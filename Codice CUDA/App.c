#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>

#include "LocalBinaryPatternGPUShared.h"
#include "LocalBinaryPatternGPUTexture.h"
#include "GrayScaleImage.h"
#include "LocalBinaryPattern.h"
#include "Histogram.h"
#include "Timer.h"
#include "RunConfiguration.h"
#include "Test.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <io.h>
#include <vector>

using namespace std;

// timing file name
const string TIMES_FILE_NAME = ".\\times.csv";

// output files are csv, in which elements are separated by CSV_SEPARATOR
const char CSV_SEPARATOR = ',';

/**
* Generate a histogram file model
* args:
*  LaunchOptions launchOptions:	options specified through command line
* return:	a string with format ".\\model_p{launchOptions.points}_r{launchOptions.radius}.csv
*/
string histogramFileName(LaunchOptions launchOptions);

/**
* Print the help menu when no args are passed to the program
*/
void printHelpMenu();

/**
* Utility function that checks if a file exists, used to append rows to time and histogram logs
* args:
*  const std::string&:	path of the file in filesystem
*/
bool fileExists(const std::string& filePath);

/**
* Create a time log file or if exists add a row.
* A row contains image informations, LBP parameters used and elapsed time.
* args:
*  LaunchOptions launchOptions:		options specified through command line
*  GrayScaleImage grayScaleImage:	the image on which LBP is applied
*  double time:						the elapsed time during algorithm execution
*/
void logTime(LaunchOptions launchOptions, GrayScaleImage grayScaleImage, double time);

/**
* Create an histogram log file or if exists add a row.
* A row contains the label of the image and frequency for each bin.
* args:
*  LaunchOptions launchOptions:	options specified through command line
*  Histogram histogram:			the resulting histogram of LBP
*/
void logHistogram(LaunchOptions launchOptions, Histogram histogram);

/**
* Program entry point
*/
int main(int argc, char** argv)
{
	// force init of CUDA context
	CHECK(cudaFree(0));

	if (argc < 2) {
		printHelpMenu();
	}
	else {
		// check command line options validity and save them
		LaunchOptions options = saveLaunchOptions(argc, argv);
		checkLaunchOptions(argc, options);

		// launch tests or LBP on a given image
		if (options.testMode) {
			testCompareWithOldHistograms(options.path.c_str());
			testLBPOnRandomImages();
		} else {
			GrayScaleImage* h_image = buildGrayScaleImage(options.path.c_str());
			if (h_image == NULL) {
				exitAfterError("Error loading image: unknown format, file not found or pixel size different than 8 bit");
			}

			// result variables
			Histogram* histogram;
			double elapsedTime = -1;

			// run LBP on one of three supports: CPU, shared kernel or texture kernel
			if (options.device == CPU_DEVICE) {
				histogram = localBinaryPatternCPUBilinear(h_image, options.points, options.radius, &elapsedTime);
			}
			else if (options.device == SHARED_DEVICE) {
				histogram = LocalBinaryPatternGPUSharedMemory(h_image, options.points, options.radius, true, &elapsedTime);
			}
			else if (options.device == TEXTURE_DEVICE) {
				histogram = LocalBinaryPatternGPUTextureMemory(h_image, options.points, options.radius, true, &elapsedTime);
			}
			
			// log execution results
			logTime(options, *h_image, elapsedTime);
			logHistogram(options, *histogram);

			// free memory
			releaseHistogram(histogram);
			releaseGrayScaleImage(h_image);
		}
	}

	CHECK(cudaDeviceReset());
	return 0;
}

string histogramFileName(LaunchOptions launchOptions) {
	stringstream stringStream;
	stringStream << ".\\model_p" << launchOptions.points << "_r" << launchOptions.radius << ".csv";
	return stringStream.str();
}

void printHelpMenu() {
	cout << "Help" << endl;
	cout << "1. Launch LBP algorithm, this creates or appends a line to files model_p{points}_r{radius}.csv and times.csv" << endl << endl;
	cout << "\tprogramName \"path\" device points radius label" << endl << endl;
	cout << "\t\"path\":\tabsolute path of the image" << endl;
	cout << "\tdevice:\twhere the algorithm is executed (" << CPU_DEVICE << ", " << SHARED_DEVICE << ", " << TEXTURE_DEVICE << ")" << endl;
	cout << "\tpoints:\tinteger number of points [1-32]" << endl;
	cout << "\tradius:\tfloat number of points [1-32]" << endl;
	cout << "\tlabel:\tcategory of the texture" << endl;
	cout << endl;
	cout << "2. Launch tests" << endl << endl;
	cout << "\tprogramName \"path\"" << endl << endl;
	cout << "\t\"path\": absolute path of the folder containing test images, see chapter 4 of documentation" << endl;
	cout << endl;
}

bool fileExists(const std::string& filePath)
{
	struct stat buf;
	if (stat(filePath.c_str(), &buf) != -1)
	{
		return true;
	}
	return false;
}

void logTime(LaunchOptions launchOptions, GrayScaleImage grayScaleImage, double time) {
	ofstream logFile;
	if (fileExists(TIMES_FILE_NAME)) {
		logFile.open(TIMES_FILE_NAME, ios::app);
	}
	else {
		// write header
		logFile.open(TIMES_FILE_NAME);
		logFile
			<< "path" << CSV_SEPARATOR
			<< "points" << CSV_SEPARATOR
			<< "radius" << CSV_SEPARATOR
			<< "device" << CSV_SEPARATOR
			<< "width" << CSV_SEPARATOR
			<< "height" << CSV_SEPARATOR
			<< "time" << endl;
	}
	// write row
	logFile
		<< launchOptions.path << CSV_SEPARATOR
		<< launchOptions.points << CSV_SEPARATOR
		<< launchOptions.radius << CSV_SEPARATOR
		<< launchOptions.device << CSV_SEPARATOR
		<< grayScaleImage.width << CSV_SEPARATOR
		<< grayScaleImage.height << CSV_SEPARATOR
		<< time << endl;
	logFile.close();
}

void logHistogram(LaunchOptions launchOptions, Histogram histogram) {
	ofstream logFile;
	string path = histogramFileName(launchOptions);

	if (fileExists(path)) {
		logFile.open(path, ios::app);
	}
	else {
		// write header
		logFile.open(path);
		logFile << "name";
		for (int p = 0; p < histogram.binsCount; p++) {
			logFile << CSV_SEPARATOR << "bin[" << p << "]";
		}
		logFile << endl;
	}
	// write row
	logFile << launchOptions.label;
	for (int i = 0; i < histogram.binsCount; i++) {
		logFile << CSV_SEPARATOR << histogram.bins[i];
	}
	logFile << endl;
	logFile.close();

}