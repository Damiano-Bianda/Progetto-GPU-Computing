#pragma once

#include "Utils.h"
#include <chrono>

/**
* Contains CPU Timer informations
* - std::chrono::time_point<std::chrono::steady_clock> start:	contains start time
* - std::chrono::time_point<std::chrono::steady_clock> end:		contains end time
*/
typedef struct _Timer {
	std::chrono::time_point<std::chrono::steady_clock> start;
	std::chrono::time_point<std::chrono::steady_clock> end;
} Timer;

/**
* Contains CUDA Timer informations
* - cudaEvent_t:	a CUDA event that records start time
* - cudaEvent_t:	a CUDA event that records end time
*/
typedef struct _TimerCuda {
	cudaEvent_t start;
	cudaEvent_t end;
} TimerCuda;

/**
* Create a timer that register time between CPU tasks
* return:	a pointer to a timer
*/
Timer* buildTimer();

/**
* Frees memory allocated for the timer
* args:
*	- Timer* timer:	a pointer to the timer that must be freed
*/
void releaseTimer(Timer* timer);

/**
* Register start time
* args:
*  - Timer* timer:	a pointer to an unusued timer, otherwise old start data will be overwritten
*/
void startClock(Timer* timer);

/**
* Register end time
* args:
*  - Timer* timer:	a pointer to an unusued timer, otherwise old end data will be overwritten
*/
void endClock(Timer* timer);

/**
* Calculate the duration in milliseconds between start and end time
* return:	a double containing the duration in milliseconds
*/
double elapsedMilliSeconds(Timer* timer);

/**
* Calculate the duration in microseconds between start and end time
* return:	a double containing the duration in microseconds
*/
double elapsedMicroSeconds(Timer* timer);

/**
* Calculate the duration in nanoseconds between start and end time
* return:	a double containing the duration in nanoseconds
*/
double elapsedNanoSeconds(Timer* timer);

/**
* Create a timer that register time between CUDA tasks on stream 0
* return:	a pointer to a timerCuda
*/
TimerCuda* buildTimerCuda();

/**
* Frees memory allocated for the timerCuda
* args:
*	- TimerCuda* timer:	a pointer to the timerCuda that must be freed
*/
void releaseTimer(TimerCuda* timer);

/**
* Register start time
* args:
*  - TimerCuda* timer:	a pointer to an unusued timerCuda, otherwise old start data will be overwritten
*/
void startClock(TimerCuda* timer);

/**
* Register end time
* args:
*  - TimerCuda* timer:	a pointer to an unusued timerCuda, otherwise old end data will be overwritten
*/
void endClock(TimerCuda* timer);

/**
* Synchronize CPU thread to the end time
* args:
*  - TimerCuda* timer:	a pointer to a timerCuda, that has been been registered end time.
*/
void synchronize(TimerCuda* timer);

/**
* Calculate the duration in milliseconds between start and end time
* return:	a float containing the duration in milliseconds
*/
float elapsedMilliSeconds(TimerCuda* timer);