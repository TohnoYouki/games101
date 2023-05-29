#include "SwapChain.hpp"

void FrameBuffer::clear(BufferTypes types)
{
    if ((types & BufferTypes::Color) == BufferTypes::Color && frame_buf != nullptr)
    {
        std::fill(frame_buf->begin(), frame_buf->end(), Eigen::Vector3f{ 0, 0, 0 });
    }
    if ((types & BufferTypes::Depth) == BufferTypes::Depth)
    {
        std::fill(depth_buf->begin(), depth_buf->end(), std::numeric_limits<float>::infinity());
    }
}

SwapChain::SwapChain(int w, int h): width(w), height(h)
{
    depth_buf.resize(w * h);
    for (int i = 0; i < SWAPCHAIN_NUM; i++) {
        frame_buf[i].resize(w * h);
    }
}

FrameBuffer SwapChain::get_next_present_framebuffer()
{
    FrameBuffer result = {
        this->width,
        this->height,
        nullptr,
        &(this->frame_buf[present_index]),
        std::unique_lock<std::mutex>(this->mutexs[present_index])
    };
    present_index = (present_index + 1) % SWAPCHAIN_NUM;
    return result;
}

FrameBuffer SwapChain::get_next_render_framebuffer()
{
    FrameBuffer result = {
        this->width,
        this->height,
        &(this->depth_buf),
        &(this->frame_buf[render_index]),
        std::unique_lock<std::mutex>(this->mutexs[render_index])
    };
    render_index = (render_index + 1) % SWAPCHAIN_NUM;
    return result;
}