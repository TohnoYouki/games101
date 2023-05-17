#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

float point_size = 1.5;
int sample_num_sqrt = 2;

std::vector<cv::Point2f> control_points;

void mouse_handler(int event, int x, int y, int flags, void *userdata) 
{
    if (event == cv::EVENT_LBUTTONDOWN && control_points.size() < 4) 
    {
        std::cout << "Left button of the mouse is clicked - position (" << x << ", "
        << y << ")" << '\n';
        control_points.emplace_back(x, y);
    }     
}

void naive_bezier(const std::vector<cv::Point2f> &points, cv::Mat &window) 
{
    auto &p_0 = points[0];
    auto &p_1 = points[1];
    auto &p_2 = points[2];
    auto &p_3 = points[3];

    for (double t = 0.0; t <= 1.0; t += 0.001) 
    {
        auto point = std::pow(1 - t, 3) * p_0 + 3 * t * std::pow(1 - t, 2) * p_1 +
                 3 * std::pow(t, 2) * (1 - t) * p_2 + std::pow(t, 3) * p_3;

        window.at<cv::Vec3b>(point.y, point.x)[2] = 255;
    }
}

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    if (control_points.size() <= 1) { return control_points[0]; }
    // TODO: Implement de Casteljau's algorithm
    std::vector<cv::Point2f> interpolated_points;
    for (uint32_t i = 0; i < control_points.size() - 1; i++) {
        interpolated_points.push_back(control_points[i] * (1 - t) + control_points[i + 1] * t);
    }
    return recursive_bezier(interpolated_points, t);
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.
    for (double t = 0.0; t <= 1.0; t += 0.001)
    {
        auto point = recursive_bezier(control_points, t);
        
        int lower_y = static_cast<int>(std::floor(point.y - point_size / 2));
        int upper_y = static_cast<int>(std::ceil(point.y + point_size / 2));
        int lower_x = static_cast<int>(std::floor(point.x - point_size / 2));
        int upper_x = static_cast<int>(std::ceil(point.x + point_size / 2));
        
        int sample_num = sample_num_sqrt * sample_num_sqrt;
        for (int y = lower_y; y < upper_y; y++) {
            for (int x = lower_x; x < upper_x; x++) {
                float color = 0;
                for (int k = 0; k < sample_num; k++) {
                    float offsetx = static_cast<float>(1 + (k / sample_num_sqrt)) / (sample_num_sqrt + 1);
                    float offsety = static_cast<float>(1 + k % sample_num_sqrt) / (sample_num_sqrt + 1);
                    cv::Point2f vector(x + offsetx - point.x, y + offsety - point.y);
                    if (vector.dot(vector) < point_size * point_size / 4) {
                        color += 255.0;
                    }
                }
                color = color / sample_num;
                color = (float)(window.at<cv::Vec3b>(y, x)[2]) + color;
                window.at<cv::Vec3b>(y, x)[2] = std::min(255.0f, color);
            }
        }
    }
}

int main() 
{
    cv::Mat window = cv::Mat(700, 700, CV_8UC3, cv::Scalar(0));
    cv::cvtColor(window, window, cv::COLOR_BGR2RGB);
    cv::namedWindow("Bezier Curve", cv::WINDOW_AUTOSIZE);

    cv::setMouseCallback("Bezier Curve", mouse_handler, nullptr);

    int key = -1;
    while (key != 27) 
    {
        for (auto &point : control_points) 
        {
            cv::circle(window, point, 3, {255, 255, 255}, 3);
        }

        if (control_points.size() == 4) 
        {
            //naive_bezier(control_points, window);
            bezier(control_points, window);

            cv::imshow("Bezier Curve", window);
            cv::imwrite("my_bezier_curve.png", window);
            key = cv::waitKey(0);

            return 0;
        }

        cv::imshow("Bezier Curve", window);
        key = cv::waitKey(20);
    }

return 0;
}
