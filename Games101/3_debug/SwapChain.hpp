#pragma once
#include <mutex>
#include <vector>
#include <Eigen/Eigen>

enum class BufferTypes
{
    Color = 1,
    Depth = 2
};

inline BufferTypes operator|(BufferTypes a, BufferTypes b)
{
    return BufferTypes((int)a | (int)b);
}

inline BufferTypes operator&(BufferTypes a, BufferTypes b)
{
    return BufferTypes((int)a & (int)b);
}

class FrameBuffer
{
public:
	int width, height;
    std::vector<float>* depth_buf;
	std::vector<Eigen::Vector3f>* frame_buf;
    std::unique_lock<std::mutex> lock;

    void clear(BufferTypes types);
};

class SwapChain
{
public:
	static const int SWAPCHAIN_NUM = 2;
    SwapChain(int w, int h);

    std::mutex mutexs[SWAPCHAIN_NUM];
    int present_index = 0;
    int render_index = 0;
    FrameBuffer get_next_present_framebuffer();
    FrameBuffer get_next_render_framebuffer();
private:
	int width, height;
	std::vector<float> depth_buf;
	std::vector<Eigen::Vector3f> frame_buf[SWAPCHAIN_NUM];
};