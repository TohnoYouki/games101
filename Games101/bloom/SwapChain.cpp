#include <iostream>
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

Eigen::Vector3f FrameBuffer::getColorLinear(float w, float h) {
    auto w_lower = std::min(std::max(static_cast<int>(std::floor(w)), 0), width - 1);
    auto w_upper = std::min(std::max(static_cast<int>(std::floor(w) + 1), 0), width - 1);
    auto h_lower = std::min(std::max(static_cast<int>(std::floor(h)), 0), height - 1);
    auto h_upper = std::min(std::max(static_cast<int>(std::floor(h) + 1), 0), height - 1);

    Eigen::Vector3f color_a = (*frame_buf)[getIndex(w_lower, h_lower)];
    Eigen::Vector3f color_b = (*frame_buf)[getIndex(w_upper, h_lower)];
    Eigen::Vector3f color_c = (*frame_buf)[getIndex(w_lower, h_upper)];
    Eigen::Vector3f color_d = (*frame_buf)[getIndex(w_upper, h_upper)];
    return color_a * (w_upper - w) * (h_upper - h) +
        color_b * (w - w_lower) * (h_upper - h) +
        color_c * (w_upper - w) * (h - h_lower) +
        color_d * (w - w_lower) * (h - h_lower);
}

static inline float cubic_weight(float x)
{
    x = std::abs(x);
    const float a = -0.5;
    float x2 = x * x, x3 = x2 * x;
    return x <= 1 ? (a + 2) * x3 - (a + 3) * x2 + 1 :
        a * x3 - 5 * a * x2 + 8 * a * x - 4 * a;
}

Eigen::Vector3f FrameBuffer::getColorBicubic(float w, float h) {
    Eigen::Vector3f result{ 0.0, 0.0,0.0 };
    auto w_lower = static_cast<int>(std::floor(w));
    auto h_lower = static_cast<int>(std::floor(h));
    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            int u = std::min(std::max(i + w_lower, 0), width - 1);
            int v = std::min(std::max(j + h_lower, 0), height - 1);
            float weight = cubic_weight(i + w_lower - w) * cubic_weight(j + h_lower - h);
            result += (*frame_buf)[getIndex(u, v)] * weight;
        }
    }
    return result;
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