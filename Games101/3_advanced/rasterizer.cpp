//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <thread>

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector4f* _v)
{
    bool e1 = (y - _v[0].y()) * (_v[1].x() - _v[0].x()) >= (_v[1].y() - _v[0].y()) * (x - _v[0].x());
    bool e2 = (y - _v[1].y()) * (_v[2].x() - _v[1].x()) >= (_v[2].y() - _v[1].y()) * (x - _v[1].x());
    bool e3 = (y - _v[2].y()) * (_v[0].x() - _v[2].x()) >= (_v[0].y() - _v[2].y()) * (x - _v[2].x());
    return (e1 && e2 && e3) || (!(e1 || e2 || e3));
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v){
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::transform_triangle(int id)
{
    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;
    Eigen::Matrix4f mvp = projection * view * model;

    for (int k = 0; k < thread_payload.orilist->size(); k++)
    {
        if (k % thread_payload.thread_num != id) { continue; }
        Triangle* t = thread_payload.orilist->operator[](k);
        Triangle* newtri = &(thread_payload.triangles[k]);
        *newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{
                (view * model * t->v[0]),
                (view * model * t->v[1]),
                (view * model * t->v[2])
        };

        std::transform(mm.begin(), mm.end(),
            thread_payload.viewpos.begin() + 3 * k, [](auto& v) {
                return v.template head<3>();
            });

        Eigen::Vector4f v[] = {
                mvp * t->v[0],
                mvp * t->v[1],
                mvp * t->v[2]
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec.x() /= vec.w();
            vec.y() /= vec.w();
            vec.z() /= vec.w();
        }

        Eigen::Matrix4f inv_trans = (view * model).inverse().transpose();
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        for (auto& vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            //screen space coordinates
            newtri->setVertex(i, v[i]);
        }

        for (int i = 0; i < 3; ++i)
        {
            //view space normal
            newtri->setNormal(i, n[i].head<3>());
        }

        newtri->setColor(0, 148, 121.0, 92.0);
        newtri->setColor(1, 148, 121.0, 92.0);
        newtri->setColor(2, 148, 121.0, 92.0);

    }
}

void rst::rasterizer::draw(std::vector<Triangle *> &TriangleList, int id) {
    draw_id = id;
    thread_payload.orilist = &(TriangleList);
    thread_payload.triangles.resize(TriangleList.size());
    thread_payload.viewpos.resize(TriangleList.size() * 3);
    thread_payload.job_type = 0;
    wake_thread_job();
    thread_payload.job_type = 1;
    wake_thread_job();
}

static Eigen::Vector3f interpolate(const Vector3f & weight, const Vector3f * attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

static Eigen::Vector2f interpolate(const Vector3f& weight, const Vector2f* attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(int id) 
{
    for (int k = 0; k < thread_payload.triangles.size(); k++)
    {
        const Triangle* t = &(thread_payload.triangles[k]);
        std::array<Eigen::Vector3f, 3> view_pos = { thread_payload.viewpos[k * 3],
                                                    thread_payload.viewpos[k * 3 + 1],
                                                    thread_payload.viewpos[k * 3 + 2] };
        auto v = t->toVector4();
        auto& a = v[0], & b = v[1], & c = v[2];

        if ((c.y() - a.y()) * (b.x() - a.x()) <= (c.x() - a.x()) * (b.y() - a.y())) { continue; }

        int lower_x = static_cast<int>(std::max(std::floor(std::min(std::min(a.x(), b.x()), c.x())), 0.0f));
        int lower_y = static_cast<int>(std::max(std::floor(std::min(std::min(a.y(), b.y()), c.y())), 0.0f));

        int upper_x = std::min(static_cast<int>(std::ceil(std::max(a.x(), std::max(b.x(), c.x())))), width);
        int upper_y = std::min(static_cast<int>(std::ceil(std::max(a.y(), std::max(b.y(), c.y())))), height);

        for (int j = lower_y; j < upper_y; j++) {
            if (j % thread_payload.thread_num != id) { continue; }
            for (int i = lower_x; i < upper_x; i++) {

                if (!insideTriangle(i + 0.5, j + 0.5, t->v)) { continue; }
                int index = (height - j) * width + i;

                auto [alpha, beta, gamma] = computeBarycentric2D(i + 0.5, j + 0.5, t->v);
                Vector3f weight(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w());
                weight = weight / (weight.x() + weight.y() + weight.z());
                float depth = v[0].z() * weight.x() + v[1].z() * weight.y() + v[2].z() * weight.z();

                if (depth > depth_buf[draw_id][index]) { continue; }

                Eigen::Vector3f color = interpolate(weight, t->color);
                Eigen::Vector3f normal = interpolate(weight, t->normal).normalized();
                Eigen::Vector2f coords = interpolate(weight, t->tex_coords);

                fragment_shader_payload payload(color, normal, coords, texture ? &*texture : nullptr);
                payload.view_pos = interpolate(weight, view_pos.data());

                depth_buf[draw_id][index] = depth;
                frame_buf[draw_id][index] = fragment_shader(payload);
            }
        }
    }
}

void rst::rasterizer::wake_thread_job()
{
    std::unique_lock<std::mutex> lock(thread_payload.mutex);
    thread_payload.begin[thread_payload.switch_id] = true;
    thread_payload.begin[(thread_payload.switch_id + 1) % 2] = false;
    thread_payload.cv.notify_all();
    thread_payload.cv.wait(lock, [&] {
        return thread_payload.finished == thread_payload.thread_num; });
    thread_payload.finished = 0;
    lock.unlock();
}

void rst::rasterizer::thread_job(int id)
{
    int switch_id = 0;
    while (!thread_payload.exit) {
        std::unique_lock<std::mutex> lock(thread_payload.mutex);
        thread_payload.cv.wait(lock, [&] { 
            return thread_payload.begin[switch_id] || thread_payload.exit; });
        lock.unlock();
        if (thread_payload.exit) { break; }

        switch (thread_payload.job_type) {
        case 0:
            transform_triangle(id);
            break;
        case 1:
            rasterize_triangle(id);
            break;
        };

        lock.lock();
        thread_payload.finished += 1;
        switch_id = (switch_id + 1) % 2;
        thread_payload.switch_id = switch_id;
        thread_payload.cv.notify_all();
        lock.unlock();
    }
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

void rst::rasterizer::clear(rst::Buffers buff, int id)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf[id].begin(), frame_buf[id].end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf[id].begin(), depth_buf[id].end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h, int thread_num) : width(w), height(h)
{
    for (int i = 0; i < BUFNUM; i++) {
        frame_buf[i].resize(w * h);
        depth_buf[i].resize(w * h);
    }

    thread_payload.thread_num = thread_num;
    texture = std::nullopt;

    for (int i = 0; i < thread_num; i++) {
        render_threads.emplace_back(&rst::rasterizer::thread_job, this, i);
    }
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-y)*width + x;
}

void rst::rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

rst::rasterizer::~rasterizer() {
    thread_payload.exit = true;
    thread_payload.cv.notify_all();
    for (int i = 0; i < thread_payload.thread_num; i++) {
        render_threads[i].join();
    }
}