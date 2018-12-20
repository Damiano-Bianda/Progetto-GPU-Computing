#pragma once

#include <string>

using namespace std;

/**
* Contains args given by command line when the program is launched
* - testMode:	specify in which mode is launched the program (see doc)
* - path:		path of a folder or image (see doc)
* - points:		number of points for the LBP algorithm
* - radius:		radius for the LBP algorithm
* - label:		label that the user assign to an image
* - device:		support used (cpu, shared memory or texture memory)
*/
typedef struct _LaunchOptions {
	bool testMode;
	string path;
	int points;
	float radius;
	string label;
	string device;
} LaunchOptions;

/**
* Check argc and argv values in order to generate a well defined LaunchOptions object
* args:
*  - int argc:		number of command line parameters
*  - char** argv:	an array of command line parameters
* return:	corrects LaunchOptions
* warning:  this methods termines the program if passed argv are incorrect
*/
LaunchOptions saveLaunchOptions(int argc, char** argv);

/**
* Check if the generated LaunchOptions respect some values defined in Utils.h
* args:
*  - int argc:						number of command line parameters
*  - LaunchOptions launchOptions:	previously generated LaunchOptions
* warning:  this methods termines the program if passed argv are incorrect
*/
void checkLaunchOptions(int argc, LaunchOptions launchOptions);