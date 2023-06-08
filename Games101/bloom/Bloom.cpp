#include "Bloom.hpp"
#include <iostream>

Bloom::Bloom(int width, int height,
	int max_downsample_num,
	float clampmax, float threshold, float softknee, float scatter,
	float bloom_intensity, float bloom_tint,
	bool high_quality, bool unity_brightness):
	width(width), height(height), 
	clampmax(clampmax), threshold(threshold), softknee(softknee),
	bloom_intensity(bloom_intensity), bloom_tint(bloom_tint),
	scatter(scatter), high_quality(high_quality), unity_brightness(unity_brightness)
{
	for (int i = 0; i < max_downsample_num; i++) {
		width = width / 2;
		height = height / 2;
		if (width <= 0 || height <= 0) break;
		datas.emplace_back(width * height);
		buffers.emplace_back(FrameBuffer{ width, height, nullptr, nullptr });
		datas.emplace_back(width * height);
		buffers.emplace_back(FrameBuffer{ width, height, nullptr, nullptr });
	}
	for (int i = 0; i < datas.size(); i++) {
		buffers[i].frame_buf = &(datas[i]);
	}
	downsample_num = buffers.size() / 2;
}

void Bloom::apply(FrameBuffer& framebuffer, JobThread& threads)
{
	JobThread::ThreadTaskPayload task_payload;
	Bloom::BloomThreadPayload data_payload;
	task_payload.payload = static_cast<void*>(&data_payload);

	data_payload.bloom = this;
	data_payload.source1 = &framebuffer;
	data_payload.target = &(buffers[0]);
	task_payload.task = prefilter;
	threads.wake_thread_job(task_payload);

	for (int i = 1; i < downsample_num; i++) {
		data_payload.source1 = &(buffers[2 * (i - 1)]);
		data_payload.target = &(buffers[2 * i + 1]);
		task_payload.task = blur_horizontal;
		threads.wake_thread_job(task_payload);
		data_payload.source1 = &(buffers[2 * i + 1]);
		data_payload.target = &(buffers[2 * i]);
		task_payload.task = blur_vertical;
		threads.wake_thread_job(task_payload);
	}
	for (int i = downsample_num - 2; i >= 0; i--) {
		data_payload.source1 = &(buffers[2 * i]);
		data_payload.source2 = i == downsample_num - 2 ?
			        &(buffers[2 * (i + 1)]) : &(buffers[2 * (i + 1) + 1]);
		data_payload.target = &(buffers[2 * i + 1]);
		task_payload.task = upsample;
		threads.wake_thread_job(task_payload);
	}
	data_payload.source1 = &(buffers[1]);
	data_payload.target = &framebuffer;
	task_payload.task = merge;
	threads.wake_thread_job(task_payload);
}

void Bloom::prefilter(int id, int thread_num, void* ppayload) {
	BloomThreadPayload* payload = static_cast<BloomThreadPayload*>(ppayload);
	Bloom* bloom = payload->bloom;
	FrameBuffer* source = payload->source1;
	FrameBuffer* target = payload->target;
	
	for (int i = 0; i < target->width; i++) {
		if (i % thread_num != id) { continue; }
		float u = (i + 0.5) * source->width / (float)target->width;
		for (int j = 0; j < target->height; j++) {
			float v = (j + 0.5) * source->height / (float)target->height;

			Eigen::Vector3f color;
			if (bloom->high_quality) {
				Eigen::Vector3f A = source->getColorLinear(u - 1.0, v - 1.0) / 255.0;
				Eigen::Vector3f B = source->getColorLinear(u, v - 1.0) / 255.0;
				Eigen::Vector3f C = source->getColorLinear(u + 1.0, v - 1.0) / 255.0;
				Eigen::Vector3f D = source->getColorLinear(u - 0.5, v - 0.5) / 255.0;
				Eigen::Vector3f E = source->getColorLinear(u + 0.5, v - 0.5) / 255.0;
				Eigen::Vector3f F = source->getColorLinear(u - 1.0, v) / 255.0;
				Eigen::Vector3f G = source->getColorLinear(u, v) / 255.0;
				Eigen::Vector3f H = source->getColorLinear(u + 1.0, v) / 255.0;
				Eigen::Vector3f I = source->getColorLinear(u - 0.5, v + 0.5) / 255.0;
				Eigen::Vector3f J = source->getColorLinear(u + 0.5, v + 0.5) / 255.0;
				Eigen::Vector3f K = source->getColorLinear(u - 1.0, v + 1.0) / 255.0;
				Eigen::Vector3f L = source->getColorLinear(u, v + 1.0) / 255.0;
				Eigen::Vector3f M = source->getColorLinear(u + 1.0, v + 1.0) / 255.0;

				color = (D + E + I + J) * 0.125;
				color += (A + B + G + F) * 0.03125;
				color += (B + C + H + G) * 0.03125;
				color += (F + G + L + K) * 0.03125;
				color += (G + H + M + L) * 0.03125;
			} else {
				color = source->getColorLinear(u, v) / 255.0;
			}
			color = color.cwiseMin(bloom->clampmax);
			float brightness = bloom->unity_brightness ? color.maxCoeff() : 
				0.2126 * color.x() + 0.7152 * color.y() + 0.0722 * color.z();
			float softness = std::clamp(brightness - bloom->threshold + bloom->softknee, 
				                        0.0f, 2 * bloom->softknee);
			softness = (softness * softness) / (4.0 * bloom->softknee + 1e-4);
			float multiplier = std::max(brightness - bloom->threshold, softness);
			multiplier /= std::max(brightness, 1e-4f);
			color = (color * multiplier).cwiseMax(0);
			
			target->getColor(i, j) = color * 255.0;
		}
	}
}

void Bloom::blur_horizontal(int id, int thread_num, void* ppayload)
{
	BloomThreadPayload* payload = static_cast<BloomThreadPayload*>(ppayload);
	Bloom* bloom = payload->bloom;
	FrameBuffer* source = payload->source1;
	FrameBuffer* target = payload->target;

	for (int i = 0; i < target->width; i++) {
		if (i % thread_num != id) { continue; }
		float u = (i + 0.5) * source->width / (float)target->width;
		for (int j = 0; j < target->height; j++) {
			float v = (j + 0.5) * source->height / (float)target->height;
			Eigen::Vector3f color = { 0.0, 0.0, 0.0 };
			float weights[9] = { 0.01621622, 0.05405405, 0.12162162,
							     0.19459459, 0.22702703, 0.19459459,
							     0.12162162, 0.05405405, 0.01621622 };
			for (int k = -4; k <= 4; k++) {
				color += source->getColorLinear(u + k * 2, v) * weights[k + 4];
			}
			target->getColor(i, j) = color;
		}
	}
}

void Bloom::blur_vertical(int id, int thread_num, void* ppayload)
{
	BloomThreadPayload* payload = static_cast<BloomThreadPayload*>(ppayload);
	Bloom* bloom = payload->bloom;
	FrameBuffer* source = payload->source1;
	FrameBuffer* target = payload->target;

	for (int i = 0; i < target->width; i++) {
		if (i % thread_num != id) { continue; }
		float u = (i + 0.5) * source->width / (float)target->width;
		for (int j = 0; j < target->height; j++) {
			float v = (j + 0.5) * source->height / (float)target->height;
			Eigen::Vector3f color = { 0.0, 0.0, 0.0 };
			float weights[5] = { 0.07027027, 0.31621622, 0.22702703, 0.31621622, 0.07027027 };
			float offsets[5] = { -3.23076923, -1.38461538, 0.0, 1.38461538, 3.23076923};
			for (int k = 0; k < 5; k++) {
				color += source->getColorLinear(u, v + offsets[k]) * weights[k];
			}
			target->getColor(i, j) = color;
		}
	}
}

static Eigen::Vector3f lerp(const Eigen::Vector3f& a, const Eigen::Vector3f& b, float w) {
	return Eigen::Vector3f { a.x() * (1 - w) + b.x() * w,
	                         a.y() * (1 - w) + b.y() * w,
	                         a.z() * (1 - w) + b.z() * w };
}

void Bloom::upsample(int id, int thread_num, void* ppayload)
{
	BloomThreadPayload* payload = static_cast<BloomThreadPayload*>(ppayload);
	Bloom* bloom = payload->bloom;
	FrameBuffer* source1 = payload->source1;
	FrameBuffer* source2 = payload->source2;
	FrameBuffer* target = payload->target;

	for (int i = 0; i < target->width; i++) {
		if (i % thread_num != id) { continue; }
		float u1 = (i + 0.5) * source1->width / (float)target->width;
		float u2 = (i + 0.5) * source2->width / (float)target->width;
		for (int j = 0; j < target->height; j++) {
			float v1 = (j + 0.5) * source1->height / (float)target->height;
			float v2 = (j + 0.5) * source2->height / (float)target->height;
			Eigen::Vector3f color1 = source1->getColorLinear(u1, v1);
			Eigen::Vector3f color2 = bloom->high_quality ? 
				source2->getColorBicubic(u2, v2) : source2->getColorLinear(u2, v2);
			target->getColor(i, j) = lerp(color1, color2, bloom->scatter);
		}
	}
}

void Bloom::merge(int id, int thread_num, void* ppayload)
{
	BloomThreadPayload* payload = static_cast<BloomThreadPayload*>(ppayload);
	Bloom* bloom = payload->bloom;
	FrameBuffer* source = payload->source1;
	FrameBuffer* target = payload->target;

	for (int i = 0; i < target->width; i++) {
		if (i % thread_num != id) { continue; }
		float u = (i + 0.5) * source->width / (float)target->width;
		for (int j = 0; j < target->height; j++) {
			float v = (j + 0.5) * source->height / (float)target->height;
			Eigen::Vector3f color = bloom->high_quality ?
				source->getColorBicubic(u, v) : source->getColorLinear(u, v);
			target->getColor(i, j) += color * bloom->bloom_intensity * bloom->bloom_tint;
		}
	}
}