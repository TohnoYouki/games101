#pragma once

#include <iostream>

class FrameCounter
{
private:
	float interval;
	float fps = 0.0f;
	int last_frame = 0;
	int now_frame = 0;
	clock_t last_time = clock();
public:
	FrameCounter(float interval) : interval(interval) {};

	float latest_fps() 
	{
		clock_t now_time = clock();
		now_frame++;
		double diff_time = double(now_time - last_time) / CLOCKS_PER_SEC;
		if (diff_time < interval) { return fps; }
		fps = (now_frame - last_frame) / diff_time;
		last_time = now_time;
		last_frame = now_frame;
		return fps;
	}
};