
#include <stdlib.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "Point.h"
#include "Utils.h"

Point* surroundingPoints(const int P, const float R) {

	Point* points = (Point*)malloc(P * sizeof(Point));
	// iterate over an angle and calculate coordinates near the circumference through circle parametric equation
	float angle = (2.0f * M_PI) / P;
	for (int p = 0; p < P; p++) {
		float arg = angle * p;
		points[p].x = roundf(-R * sinf(arg));
		points[p].y = roundf(R * cosf(arg));
	}

	return points;
}

/*
* in case we have a float between [-1e-6f,1e-6f] convert to 0
* args:
*  float value: value to check and if necessary convert
*/
float absNegativeZero(float value) {
	float integral;
	if (fabsf(modff(value, &integral)) < 1e-6f) {
		if (integral == -0.0f) {
			return fabsf(integral);
		}
		return integral;
	}
	return value;
}

PointF* surroundingPointsF(const int P, const float R) {

	PointF* points = (PointF*)malloc(P * sizeof(PointF));

	// iterate over an angle and calculate coordinates over the circumference through circle parametric equation
	float angle = (2.0 * M_PI) / P;
	for (int p = 0; p < P; p++) {
		float arg = angle * p;
		points[p].x = absNegativeZero(-R * sinf(arg));
		points[p].y = absNegativeZero(R * cosf(arg));
	}

	return points;
}

void surroundingBilinearPoints(const int points, const float radius, Point** bilinearNeighborhoodCoordinates, PointF** bilinearParameters) {
	
	PointF* pointsOverCircumference = surroundingPointsF(points, radius);
	*bilinearNeighborhoodCoordinates = (Point*) malloc(points * sizeof(Point));
	*bilinearParameters = (PointF*)malloc(points * sizeof(PointF));

	for (int p = 0; p < points; p++){
		// given a pixel over the circunference find nearest top left integer coordinates
		(*bilinearNeighborhoodCoordinates)[p].x = floorf(pointsOverCircumference[p].x + 1e-6f);
		(*bilinearNeighborhoodCoordinates)[p].y = floorf(pointsOverCircumference[p].y + 1e-6f);

		// calculate alfa and beta distance
		(*bilinearParameters)[p].x = fabsf(pointsOverCircumference[p].x - (*bilinearNeighborhoodCoordinates)[p].x);
		(*bilinearParameters)[p].y = fabsf(pointsOverCircumference[p].y - (*bilinearNeighborhoodCoordinates)[p].y);
	}

	free(pointsOverCircumference);
}

void releasePoint(Point* point) {
	free(point);
}

void releasePoint(PointF* point) {
	free(point);
}