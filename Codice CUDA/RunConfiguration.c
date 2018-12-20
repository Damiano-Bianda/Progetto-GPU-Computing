
#include "RunConfiguration.h"
#include "Utils.h"


bool isStringInVector(string str, string vec[], int size) {
	for (int i = 0; i < size; i++) {
		string test = vec[i];
		if (vec[i] == str) {
			return true;
		}
	}
	return false;
}

LaunchOptions saveLaunchOptions(int argc, char** argv) {

	if (argc != 2 && argc != 6) {
		exitAfterError("Correct launch options are required, launch program with no args and see instructions");
	}

	LaunchOptions launchOptions;

	launchOptions.path = argv[1];
	
	if (argc == 6) {
		launchOptions.testMode = false;
		launchOptions.device = string(argv[2]);
		try {
			launchOptions.points = stoi(argv[3]);
		}
		catch (invalid_argument ex) {
			exitAfterError("Use a valid int for points option");
		}
		catch (out_of_range ex) {
			exitAfterError("Use a valid int for points option");
		}
		try {
			launchOptions.radius = stof(argv[4]);
		}
		catch (invalid_argument ex) {
			exitAfterError("Use a valid float for radius option");
		}
		catch (out_of_range ex) {
			exitAfterError("Use a valid float for radius option");
		}
		launchOptions.label = string(argv[5]);
	}
	else if (argc == 2){
		launchOptions.testMode = true;
		launchOptions.device = "";
		launchOptions.points = 0;
		launchOptions.radius = 0;
		launchOptions.label = "";
	}

	return launchOptions;
}

void checkLaunchOptions(int argc, LaunchOptions launchOptions) {
	if (argc == 6) {
		string devices[] = { SHARED_DEVICE, CPU_DEVICE, TEXTURE_DEVICE };
		if (!isStringInVector(launchOptions.device, devices, 3)) {
			exitAfterError("Unknown device option: " + launchOptions.device);
		}
		if (launchOptions.points < MIN_POINTS || launchOptions.points > MAX_POINTS) {
			exitAfterError("Points option must be in integer range [" + to_string(MIN_POINTS) + ", " + to_string(MAX_POINTS) + "]");
		}
		if (launchOptions.radius < MIN_RADIUS || launchOptions.radius > MAX_RADIUS) {
			exitAfterError("Radius option must be in float range [" + to_string(MIN_RADIUS) + ", " + to_string(MAX_RADIUS) + "]");
		}
	}
}