#pragma once

/**
* Contains Point informations
* - x:	integer value of x
* - y:	integer value of y
*/
typedef struct _Point {
	int x;
	int y;
} Point;

/**
* Contains PointF informations
* - x:	float value of x
* - y:	float value of y
*/
typedef struct _PointF {
	float x;
	float y;
} PointF;

/**
* Create an array of P Point, each of them located near a circumference of radius R with center in (0,0).
* It is used for Nearest Neighbour coordinates, so Point coordinates are the nearest integer value to the circumference.
* args:
*  const int P:	number of points
*  const int R: length of radius
* return:	a 1D array of Point
*/
Point* surroundingPoints(const int P, const float R);

/**
* Create an array of P PointF, each of them located over a circumference of radius R with center in (0,0)
* args:
*  const int P:	number of points
*  const int R: length of radius
* return:	a 1D array of PointF
*/
PointF* surroundingPointsF(const int P, const float R);

/**
* Create two arrays:
* 1 Create an array of points Point, those are the top-left points used during bilinear interpolation calculation
* 2 Create an array of points PointF, those are the distances alfa e beta used during bilinear interpolation calculation
* args:
*  const int points:						number of points
*  const int radius:						length of radius
*  Point** bilinearNeighborhoodCoordinates:	a double pointer used as return value
*  PointF** bilinearParameters:				a double pointer used as return value
*/
void surroundingBilinearPoints(const int points, const float radius, Point** bilinearNeighborhoodCoordinates, PointF** bilinearParameters);

/**
* Frees memory allocated for the array of Point
* args:
*	- Point* point:	a pointer to array of Point that must be freed
*/
void releasePoint(Point* point);

/**
* Frees memory allocated for the array of PointF
* args:
*	- Point* point:	a pointer to array of PointF that must be freed
*/
void releasePoint(PointF* point);