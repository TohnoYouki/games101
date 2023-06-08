#pragma once
#include <vector>
#include <Eigen/Eigen>
#include "JobThread.hpp"
#include "SwapChain.hpp"

class Bloom
{
private:
	std::vector<std::vector<Eigen::Vector3f>> datas;
public:
	int width, height;
	int downsample_num;
	bool high_quality, unity_brightness;
	float bloom_intensity, bloom_tint;
	float clampmax, threshold, softknee, scatter;
	std::vector<FrameBuffer> buffers;

	bool open = false;

    struct BloomThreadPayload {
        Bloom* bloom;
        FrameBuffer* source1;
		FrameBuffer* source2;
		FrameBuffer* target;
    };

	Bloom(int width, int height, 
		  int max_downsample_num,
		  float clampmax, float threshold, float softknee, float scatter,
		  float bloom_intensity, float bloom_tint,
		  bool high_quality, bool unity_brightness);

	void apply(FrameBuffer& framebuffer, JobThread& threads);

private:
	static void prefilter(int id, int thread_num, void* ppayload);
	static void blur_horizontal(int id, int thread_num, void* ppayload);
	static void blur_vertical(int id, int thread_num, void* ppayload);
	static void upsample(int id, int thread_num, void* ppayload);
	static void merge(int id, int thread_num, void* ppayload);
};