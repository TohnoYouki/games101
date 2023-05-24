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

static Eigen::Vector3f interpolate(const Vector3f& weight, const Vector3f* attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

static Eigen::Vector2f interpolate(const Vector3f& weight, const Vector2f* attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

static bool insideTriangle(float x, float y, const Vector4f* _v)
{
    bool e1 = (y - _v[0].y()) * (_v[1].x() - _v[0].x()) >= (_v[1].y() - _v[0].y()) * (x - _v[0].x());
    bool e2 = (y - _v[1].y()) * (_v[2].x() - _v[1].x()) >= (_v[2].y() - _v[1].y()) * (x - _v[1].x());
    bool e3 = (y - _v[2].y()) * (_v[0].x() - _v[2].x()) >= (_v[0].y() - _v[2].y()) * (x - _v[2].x());
    return (e1 && e2 && e3) || (!(e1 || e2 || e3));
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector4f* v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return { c1,c2,c3 };
}

rst::JobThread::JobThread(int num)
{
    control_payload.thread_num = num;
    for (int i = 0; i < num; i++) {
        threads.emplace_back(&rst::JobThread::thread_job, this, i);
    }
}

rst::JobThread::~JobThread() {
    control_payload.exit = true;
    control_payload.cv.notify_all();
    for (int i = 0; i < control_payload.thread_num; i++) {
        threads[i].join();
    }
}

void rst::JobThread::wake_thread_job(const ThreadTaskPayload& payload)
{
    task_payload = &(payload);
    std::unique_lock<std::mutex> lock(control_payload.mutex);
    control_payload.wake_semaphore[control_payload.wake_semaphore_index] = true;
    control_payload.wake_semaphore[(control_payload.wake_semaphore_index + 1) % 2] = false;

    control_payload.cv.notify_all();
    control_payload.cv.wait(lock, [&] {
        return control_payload.finished_thread == control_payload.thread_num; });
    control_payload.finished_thread = 0;
    lock.unlock();
}

void rst::JobThread::thread_job(int id)
{
    int wake_semaphore_index = 0;
    while (!control_payload.exit) {
        std::unique_lock<std::mutex> lock(control_payload.mutex);

        control_payload.cv.wait(lock, [&] {
            return control_payload.wake_semaphore[wake_semaphore_index] 
                || control_payload.exit; });
        lock.unlock();
        if (control_payload.exit) { break; }

        task_payload->task(id, control_payload.thread_num, task_payload->payload);

        lock.lock();

        control_payload.finished_thread += 1;
        wake_semaphore_index = (wake_semaphore_index + 1) % 2;
        control_payload.wake_semaphore_index = wake_semaphore_index;
        control_payload.cv.notify_all();

        lock.unlock();
    }
}

void rst::Rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::Rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::Rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::Rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void rst::Rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

void rst::Rasterizer::clear(rst::Buffers buff, int id)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf[id].begin(), frame_buf[id].end(), Eigen::Vector3f{ 0, 0, 0 });
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::Rasterizer::Rasterizer(int w, int h) : width(w), height(h)
{
    depth_buf.resize(w * h);
    for (int i = 0; i < SWAPCHAIN_NUM; i++) {
        frame_buf[i].resize(w * h);
    }
    texture = std::nullopt;
}

void rst::Rasterizer::transform_triangle(int id, int thread_num, void * ppayload)
{
    RasterizerThreadPayload* payload = static_cast<RasterizerThreadPayload*>(ppayload);
    Rasterizer* r = payload->rasterizer;

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;
    Eigen::Matrix4f mvp = r->projection * r->view * r->model;

    for (int k = 0; k < payload->model_triangles->size(); k++)
    {
        if (k % thread_num != id) { continue; }
        Triangle* t = payload->model_triangles->operator[](k);
        Triangle* newtri = &(payload->screen_triangles[k]);
        *newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{
                (r->view * r->model * t->v[0]),
                (r->view * r->model * t->v[1]),
                (r->view * r->model * t->v[2])
        };

        std::transform(mm.begin(), mm.end(),
            payload->viewpos.begin() + 3 * k, [](auto& v) {
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

        Eigen::Matrix4f inv_trans = (r->view * r->model).inverse().transpose();
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        for (auto& vert : v)
        {
            vert.x() = 0.5 * r->width * (vert.x() + 1.0);
            vert.y() = 0.5 * r->height * (vert.y() + 1.0);
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

void rst::Rasterizer::rasterize_triangle(int id, int thread_num, void * ppayload)
{
    RasterizerThreadPayload* payload = static_cast<RasterizerThreadPayload*>(ppayload);
    Rasterizer* r = payload->rasterizer;

    for (int k = 0; k < payload->screen_triangles.size(); k++)
    {
        const Triangle* t = &(payload->screen_triangles[k]);
        std::array<Eigen::Vector3f, 3> view_pos = { payload->viewpos[k * 3],
                                                    payload->viewpos[k * 3 + 1],
                                                    payload->viewpos[k * 3 + 2] };
        auto v = t->toVector4();
        auto& a = v[0], & b = v[1], & c = v[2];
        if ((c.y() - a.y()) * (b.x() - a.x()) <= (c.x() - a.x()) * (b.y() - a.y())) { continue; }

        int lower_x = static_cast<int>(std::max(std::floor(std::min(std::min(a.x(), b.x()), c.x())), 0.0f));
        int lower_y = static_cast<int>(std::max(std::floor(std::min(std::min(a.y(), b.y()), c.y())), 0.0f));

        int upper_x = std::min(static_cast<int>(std::ceil(std::max(a.x(), std::max(b.x(), c.x())))), r->width);
        int upper_y = std::min(static_cast<int>(std::ceil(std::max(a.y(), std::max(b.y(), c.y())))), r->height);

        for (int j = lower_y; j < upper_y; j++) {
            if (j % thread_num != id) { continue; }
            for (int i = lower_x; i < upper_x; i++) {

                if (!insideTriangle(i + 0.5, j + 0.5, t->v)) { continue; }
                int index = (r->height - j) * r->width + i;

                auto [alpha, beta, gamma] = computeBarycentric2D(i + 0.5, j + 0.5, t->v);
                Vector3f weight(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w());
                weight = weight / (weight.x() + weight.y() + weight.z());
                if (isnan(weight.x()) || isnan(weight.y()) || isnan(weight.z())) { continue; }
                float depth = v[0].z() * weight.x() + v[1].z() * weight.y() + v[2].z() * weight.z();

                if (depth > r->depth_buf[index]) { continue; }

                Eigen::Vector3f color = interpolate(weight, t->color);
                Eigen::Vector3f normal = interpolate(weight, t->normal).normalized();
                Eigen::Vector2f coords = interpolate(weight, t->tex_coords);

                fragment_shader_payload fragment_payload(
                    color, normal, coords, r->texture ? &*(r->texture) : nullptr);
                fragment_payload.view_pos = interpolate(weight, view_pos.data());

                r->depth_buf[index] = depth;
                r->frame_buf[payload->draw_id][index] = r->fragment_shader(fragment_payload);
            }
        }
    }
}

void rst::Rasterizer::draw(std::vector<Triangle*>& TriangleList, int id, rst::JobThread & threads) {
    rst::JobThread::ThreadTaskPayload task_payload;
    RasterizerThreadPayload data_payload;
    data_payload.draw_id = id;
    data_payload.rasterizer = this;
    data_payload.model_triangles = &(TriangleList);
    data_payload.screen_triangles.resize(TriangleList.size());
    data_payload.viewpos.resize(TriangleList.size() * 3);
    task_payload.payload = static_cast<void *>(&data_payload);
    task_payload.task = transform_triangle;
    threads.wake_thread_job(task_payload);
    task_payload.task = rasterize_triangle;
    threads.wake_thread_job(task_payload);
}

//Screen space rasterization
