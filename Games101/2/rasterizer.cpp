// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{
    bool e1 = (y - _v[0].y()) * (_v[1].x() - _v[0].x()) >= (_v[1].y() - _v[0].y()) * (x - _v[0].x());
    bool e2 = (y - _v[1].y()) * (_v[2].x() - _v[1].x()) >= (_v[2].y() - _v[1].y()) * (x - _v[1].x());
    bool e3 = (y - _v[2].y()) * (_v[0].x() - _v[2].x()) >= (_v[0].y() - _v[2].y()) * (x - _v[2].x());
    return (e1 && e2 && e3) || (!(e1 || e2 || e3));
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1f) / 2.0f;
    float f2 = (50 + 0.1f) / 2.0f;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5f*width*(vert.x()+1.0f);
            vert.y() = 0.5f*height*(vert.y()+1.0f);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    auto& a = v[0], & b = v[1], & c = v[2];

    int lower_x = static_cast<int>(std::max(std::floor(std::min(std::min(a.x(), b.x()), c.x())), 0.0f));
    int lower_y = static_cast<int>(std::max(std::floor(std::min(std::min(a.y(), b.y()), c.y())), 0.0f));

    int upper_x = std::min(static_cast<int>(std::ceil(std::max(a.x(), std::max(b.x(), c.x())))), width);
    int upper_y = std::min(static_cast<int>(std::ceil(std::max(a.y(), std::max(b.y(), c.y())))), height);
    
    for (int j = lower_y; j < upper_y; j++) {
        for (int i = lower_x; i < upper_x; i++) {   
            int sample_count = sample_count_sqrt * sample_count_sqrt;

            for (int k = 0; k < sample_count; k++) {
                float offsetx = static_cast<float>(1 + (k / sample_count_sqrt)) / (sample_count_sqrt + 1);
                float offsety = static_cast<float>(1 + k % sample_count_sqrt) / (sample_count_sqrt + 1);

                if (!insideTriangle(i + offsetx, j + offsety, t.v)) { continue; }
   
                auto [alpha, beta, gamma] = computeBarycentric2D(i + offsetx, j + offsety, t.v);

                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                Eigen::Vector3f color = (t.color[0] * alpha / v[0].w() + t.color[1] * beta / v[1].w() + t.color[2] * gamma / v[2].w());
                color = color * w_reciprocal;

                if (z_interpolated <= depth_buf[get_index(i, j) * sample_count + k]) {
                    depth_buf[get_index(i, j) * sample_count + k] = z_interpolated;
                    super_frame_buf[get_index(i, j) * sample_count + k] = color * 255.0;
                }
            }      
        }
    }
    
}

std::vector<Eigen::Vector3f>& rst::rasterizer::frame_buffer() 
{ 
    int sample_count = sample_count_sqrt * sample_count_sqrt;
    for (int i = 0; i < frame_buf.size(); i++)
    {   
        frame_buf[i] = super_frame_buf[sample_count * i];
        for (int j = 1; j < sample_count; j++) {
            frame_buf[i] += super_frame_buf[sample_count * i + j];
        }
        frame_buf[i] /= sample_count;
    }
    return frame_buf; 
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(super_frame_buf.begin(), super_frame_buf.end(), Eigen::Vector3f{ 0, 0, 0 });
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    int sample_count = sample_count_sqrt * sample_count_sqrt;
    frame_buf.resize(w * h);
    super_frame_buf.resize(w * h * sample_count);
    depth_buf.resize(w * h * sample_count);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on