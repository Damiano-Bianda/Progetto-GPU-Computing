#include "Test.h"

#include "LocalBinaryPattern.h"
#include "LocalBinaryPatternGPUShared.h"
#include "LocalBinaryPatternGPUTexture.h"
#include "GrayScaleImage.h"
#include <vector>

bool isEqual(Histogram* h1, Histogram* h2) {
	if (h1->binsCount != h2->binsCount) {
		return false;
	}
	for (int i = 0; i < h1->binsCount; i++) {
		if (h1->bins[i] != h2->bins[i]) {
			return false;
		}
	}
	return true;
}

unsigned int countElementInHistogram(Histogram* h) {
	unsigned int total = 0;
	for (int i = 0; i < h->binsCount; i++) {
		total += h->bins[i];
	}
	return total;
}


Histogram* getCorrectSharedHistogram(unsigned int rowIndex, unsigned int points) {

	Histogram* histogram = nullptr;
	unsigned int bins = points + 2;
	if (points == 8) {
		histogram = buildHistogram(bins);
		std::vector<unsigned int> values =
		{ 10658, 13291, 5781, 11263, 14666, 14554, 8268, 14921, 23045, 27553,
			56352, 56450, 21247, 35319, 41994, 43516, 27664, 62806, 104406, 126246,
			285413, 232321, 81745, 119321, 125231, 134384, 97054, 245405, 443125, 540001,
			776876, 940828, 434377, 662301, 712698, 726010, 632556, 990749, 1418885, 1920720,
			808237, 2117480, 1453668, 4955593, 5871690, 6153277, 3733480, 3041842, 4229569, 4499164 };
		for (int i = 0; i < bins; i++) {
			histogram->bins[i] = values[rowIndex * bins + i];
		}
	}
	else if (points == 16) {
		histogram = buildHistogram(bins);
		std::vector<unsigned int> values =
		{ 5943,7129,2753,2357,2101,3111,4144,6573,9233,8264,4547,3548,2573,3020,4142,6805,12129,55628,
			32450,35358,11309,7537,5864,7470,9136,15516,21509,18905,10278,8340,7036,9510,15642,35937,59340,264863,
			169673,160482,41950,22963,16991,19142,21284,37985,52903,47524,22959,19649,17899,26911,51905,166249,261137,1146394,
			859949,554638,261939,129294,87944,95373,97761,153643,188019,181057,93528,83876,87779,144132,299463,566653,1151577,4179375,
			2217292,1189701,1263994,1286972,1019221,1152672,1247908,1656631,1835395,1789745,1227710,1227429,1247153,1571656,1572223,1156758,3540549,10660991 };
		for (int i = 0; i < bins; i++) {
			histogram->bins[i] = values[rowIndex * bins + i];
		}
	}
	else if (points == 24) {
		histogram = buildHistogram(bins);
		std::vector<unsigned int> values =
		{ 4604,4537,1731,1264,1092,1202,1321,1748,2168,2818,3502,5473,6874,6420,3928,2924,1939,1714,1114,1189,1187,1599,2321,4280,8140,68911,
			24797,24708,7774,4473,3427,2954,2875,3491,4103,5621,7375,11571,14794,13264,7607,5996,4166,3708,3014,3338,3611,5380,9627,25316,42663,330347,
			137522,119236,29487,14793,9975,7896,6600,7343,7960,11055,13477,23352,31472,28023,14078,11172,7921,7795,6734,7510,9048,15719,32956,125730,199289,1417857,
			567816,427597,172113,77013,45955,36033,31049,33966,34391,40662,46880,79653,103830,96816,49693,38719,28728,28936,27324,33593,43487,85397,194665,450342,767486,5673856,
			2722232,984065,999338,841787,617746,462701,346038,351982,359641,389422,438877,643131,743262,725088,432612,344194,305320,326648,353432,484755,682175,1011263,1160279,936049,3622032,16579931 };
		for (int i = 0; i < bins; i++) {
			histogram->bins[i] = values[rowIndex * bins + i];
		}
	}
	return histogram;
}

Histogram* getCorrectTextureHistogram(unsigned int rowIndex, unsigned int points) {

	Histogram* histogram = nullptr;
	unsigned int bins = points + 2;
	if (points == 8) {
		histogram = buildHistogram(bins);
		std::vector<unsigned int> values =
		{ 10658, 13231, 5792, 11362, 14730, 14611, 8432, 15000, 23404, 26780,
			56352, 56236, 21308, 35571, 42180, 43664, 27887, 63266, 105260, 124276,
			285413, 231636, 81993, 119933, 125613, 134682, 97128, 246999, 444410, 536193,
			776876, 936487, 436106, 664512, 715277, 728359, 634797, 994078, 1428844, 1900664,
			808237, 2092811, 1448285, 4980749, 5918107, 6210482, 3819746, 2982831, 4419717, 4183035 };
		for (int i = 0; i < bins; i++) {
			histogram->bins[i] = values[rowIndex * bins + i];
		}
	}
	else if (points == 16) {
		histogram = buildHistogram(bins);
		std::vector<unsigned int> values =
		{ 5937,7101,2755,2356,2109,3100,4147,6567,9289,8287,4572,3583,2588,3055,4199,6810,12238,55307,
			32427,35241,11363,7556,5870,7466,9146,15487,21567,18969,10297,8361,7080,9537,15710,36057,59565,264301,
			169669,160195,42029,23012,17013,19154,21267,38028,52960,47661,22988,19738,17963,26869,51896,166669,261504,1145385,
			859553,552946,262414,129698,88196,95483,97880,153863,188345,181476,93729,84254,88094,144548,299635,567941,1154491,4173454,
			2213383,1164079,1276281,1287785,1023591,1157041,1251768,1660805,1844325,1802021,1237012,1241915,1264504,1592550,1597827,1125325,3609002,10514786 };
		for (int i = 0; i < bins; i++) {
			histogram->bins[i] = values[rowIndex * bins + i];
		}
	}
	else if (points == 24) {
		histogram = buildHistogram(bins);
		std::vector<unsigned int> values =
		{ 4603,4558,1739,1255,1111,1197,1321,1763,2198,2794,3538,5455,6865,6406,3951,2927,1947,1702,1111,1182,1223,1584,2378,4183,8130,68879,
			24799,24772,7784,4464,3417,2935,2864,3497,4122,5575,7387,11527,14772,13256,7635,5960,4166,3699,3020,3333,3647,5384,9769,24983,42533,330700,
			137052,119336,29315,14735,9944,7883,6601,7336,7953,11040,13486,23309,31409,28072,14057,11165,7861,7772,6724,7519,9077,15794,32959,125199,198500,1419902,
			566696,427917,171537,76893,45837,35991,31015,33947,34388,40602,46866,79619,103692,96815,49667,38740,28708,28915,27311,33511,43606,85175,194865,448745,765293,5679649,
			2725254,983611,1004658,837538,620061,463141,346545,352413,361807,389080,439928,643428,743783,725592,434206,343857,306346,327202,354433,484631,686110,1010046,1171630,921865,3622553,16564282 };
		for (int i = 0; i < bins; i++) {
			histogram->bins[i] = values[rowIndex * bins + i];
		}
	}
	return histogram;
}


void testCompareWithOldHistograms(const char* folderPath) {

	std::cout << "test suite 1 - Compares CPU, Shared and Texture results with old correct histograms" << std::endl;

	std::vector<std::string> imagePaths = {
		"480x300.jpg",
		"960x600.jpg",
		"1920x1200.jpg",
		"3840x2400.jpg",
		"7680x4800.jpg"
	};
	std::vector<unsigned int> sizes = { 480 * 300, 960 * 600, 1920 * 1200, 3840 * 2400, 7680 * 4800 };
	std::vector<unsigned int> points = { 8, 16, 24 };
	std::vector<float> radiuses = { 1, 2, 3 };

	for (int dev = 0; dev < 3; dev++) {
		for (int i = 0; i < imagePaths.size(); i++) {
			std::string completePath = string(folderPath) + imagePaths[i];
			GrayScaleImage* image = buildGrayScaleImage(completePath.c_str());
			if (image == NULL) {
				std::cout << "Image at path \"" << completePath << "\" can not be loaded, test is skipped" << std::endl;
				continue;
			}
			for (int j = 0; j < points.size(); j++) {
				double elapsedTime = -1;
				Histogram* histogram;
				Histogram* correctHistogram;
				if (dev == 0) {
					histogram = localBinaryPatternCPUBilinear(image, points[j], radiuses[j], &elapsedTime);
					correctHistogram = getCorrectSharedHistogram(i, points[j]);
				}
				else if (dev == 1) {
					histogram = LocalBinaryPatternGPUSharedMemory(image, points[j], radiuses[j], true, &elapsedTime);
					correctHistogram = getCorrectSharedHistogram(i, points[j]);
				}
				else if (dev == 2) {
					histogram = LocalBinaryPatternGPUTextureMemory(image, points[j], radiuses[j], true, &elapsedTime);
					correctHistogram = getCorrectTextureHistogram(i, points[j]);
				}

				bool passed = true;
				if (countElementInHistogram(histogram) != sizes[i]) {
					std::cout << "Test failed: histogram count is different than width*height" << std::endl;
					passed = false;
				}
				if (!isEqual(histogram, correctHistogram)) {
					std::cout << "Test failed: correct histogram and new don't match" << std::endl;
					passed = false;
				}
				if (passed) {
					std::cout << "Test passed\t";
				}
				std::cout << (dev == 0 ? "cpu" : (dev ==  1 ? "shared" : "texture")) << "\t";
				std::cout << "path: " << completePath << "\t";
				std::cout << "p = " << points[j] << "\t";
				std::cout << "r = " << radiuses[j] << "\t";
				std::cout << "time = " << elapsedTime << std::endl;

				releaseHistogram(histogram);
				releaseHistogram(correctHistogram);
			}
			releaseGrayScaleImage(image);
		}
	}

	std::cout << "Test finished" << std::endl;
}

void testLBPOnRandomImages() {

	std::cout << "Test suite 2 - Generates random images, checks if histograms have right size and match between CPU and Shared" << std::endl;

	std::vector<unsigned int> imageSizes = {
		480, 300,
		960, 600,
		1920, 1200,
		3840, 2400,
		7680, 4800
	};
	std::vector<unsigned int> points = { 8, 16, 24 };
	std::vector<float> radiuses = { 1, 2, 3 };

	for (int i = 0; i < imageSizes.size(); i += 2) {
		unsigned int width = imageSizes[i];
		unsigned int height = imageSizes[i + 1];
		unsigned int pixelsCount = width * height;
		GrayScaleImage* image = createFakeImage(width, height);
		for (int j = 0; j < points.size(); j++) {
			double elapsedTime = -1;
			Histogram* histogramCPU = localBinaryPatternCPUBilinear(image, points[j], radiuses[j], &elapsedTime);
			Histogram* histogramShared = LocalBinaryPatternGPUSharedMemory(image, points[j], radiuses[j], true, &elapsedTime);
			Histogram* histogramTexture = LocalBinaryPatternGPUTextureMemory(image, points[j], radiuses[j], true, &elapsedTime);
			bool passed = true;
			if (countElementInHistogram(histogramCPU) != pixelsCount) {
				std::cout << "Test failed: CPU histogram count is different than width*height" << std::endl;
				passed = false;
			}
			if (!countElementInHistogram(histogramShared)) {
				std::cout << "Test failed: shared histogram count is different than width*height" << std::endl;
				passed = false;
			}
			if (!countElementInHistogram(histogramTexture)) {
				std::cout << "Test failed: texture histogram count is different than width*height" << std::endl;
				passed = false;
			}
			if (!isEqual(histogramCPU, histogramShared)) {
				std::cout << "Test failed: CPU and shared histogram doesn't match\t";
				passed = false;
			}
			if (passed) {
				std::cout << "Test passed\t";
			}
			std::cout << "size: " << imageSizes[i] << "x" << imageSizes[i + 1] << "\t";
			std::cout << "p = " << points[j] << "\t";
			std::cout << "r = " << radiuses[j] << std::endl;

			releaseHistogram(histogramCPU);
			releaseHistogram(histogramShared);
			releaseHistogram(histogramTexture);
		}
		releaseGrayScaleImage(image);
	}

	std::cout << "Test finished" << std::endl;
}