#include "Timer.h"

Timer* buildTimer() {
	return (Timer*)malloc(sizeof(Timer));
}

void releaseTimer(Timer* timer) {
	free(timer);
}

void startClock(Timer* timer) {
	timer->start = std::chrono::high_resolution_clock::now();
}

void endClock(Timer* timer) {
	timer->end = std::chrono::high_resolution_clock::now();
}

double elapsedMilliSeconds(Timer* timer) {
	return elapsedMicroSeconds(timer) / 1000;
}

double elapsedMicroSeconds(Timer* timer) {
	return elapsedNanoSeconds(timer) / 1000;
}

double elapsedNanoSeconds(Timer* timer) {
	double time = std::chrono::duration_cast<std::chrono::nanoseconds>(timer->end - timer->start).count();
	return time;
}

TimerCuda* buildTimerCuda() {
	TimerCuda* timerCuda = (TimerCuda*)malloc(sizeof(TimerCuda));
	CHECK(cudaEventCreate(&(timerCuda->start)));
	CHECK(cudaEventCreate(&(timerCuda->end)));
	return timerCuda;
}

void releaseTimer(TimerCuda* timer) {
	CHECK(cudaEventDestroy(timer->start));
	CHECK(cudaEventDestroy(timer->end));
	free(timer);
}

void startClock(TimerCuda* timer) {
	CHECK(cudaEventRecord(timer->start, 0));
}

void endClock(TimerCuda* timer) {
	CHECK(cudaEventRecord(timer->end, 0));
}

void synchronize(TimerCuda* timer) {
	CHECK(cudaEventSynchronize(timer->end));
}

float elapsedMilliSeconds(TimerCuda* timer) {
	float milliseconds = 0;
	CHECK(cudaEventElapsedTime(&milliseconds, timer->start, timer->end));
	return milliseconds;
}