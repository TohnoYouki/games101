//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_TEXTURE_H
#define RASTERIZER_TEXTURE_H
#include "global.hpp"
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
class Texture{
private:
    cv::Mat image_data;

public:
    Texture(const std::string& name)
    {
        image_data = cv::imread(name);
        cv::cvtColor(image_data, image_data, cv::COLOR_RGB2BGR);
        width = image_data.cols;
        height = image_data.rows;
    }

    int width, height;

    Eigen::Vector3f getColor(float u, float v)
    {
        u = std::min(std::max(u, 0.0f), 1.0f);
        v = std::min(std::max(v, 0.0f), 1.0f);
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto color = image_data.at<cv::Vec3b>(v_img, u_img);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

    Eigen::Vector3f getColorBilinear(float u, float v)
    {
        auto u_img = u * width;
        auto v_img = (1 - v) * height;
        auto u_lower = std::min(std::max(static_cast<int>(std::floor(u_img)), 0), width);
        auto u_upper = std::min(std::max(static_cast<int>(std::ceil(u_img)), 0), width);
        auto v_lower = std::min(std::max(static_cast<int>(std::floor(v_img)), 0), height);
        auto v_upper = std::min(std::max(static_cast<int>(std::ceil(v_img)), 0), height);

        auto color_a = image_data.at<cv::Vec3b>(v_lower, u_lower);
        auto color_b = image_data.at<cv::Vec3b>(v_lower, u_upper);
        auto color_c = image_data.at<cv::Vec3b>(v_upper, u_lower);
        auto color_d = image_data.at<cv::Vec3b>(v_upper, u_upper);
        auto color = color_a * (u_upper - u_img) * (v_upper - v_img) +
                     color_b * (u_upper - u_img) * (v_img - v_lower) +
                     color_c * (u_img - u_lower) * (v_upper - v_img) +
                     color_d * (u_img - u_lower) * (v_img - v_lower);
        return Eigen::Vector3f(color[0], color[1], color[2]);
    }

};
#endif //RASTERIZER_TEXTURE_H
