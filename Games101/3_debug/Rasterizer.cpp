//
// Created by goksu on 4/6/19.
//

#include <math.h>
#include "Scene.hpp"
#include "Rasterizer.hpp"

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Eigen::Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static Eigen::Vector3f interpolate(const Eigen::Vector3f& weight, const Eigen::Vector3f* attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

static Eigen::Vector2f interpolate(const Eigen::Vector3f& weight, const Eigen::Vector2f* attribute)
{
    return attribute[0] * weight.x() + attribute[1] * weight.y() + attribute[2] * weight.z();
}

static bool insideTriangle(float x, float y, const Eigen::Vector4f* _v)
{
    bool e1 = (y - _v[0].y()) * (_v[1].x() - _v[0].x()) >= (_v[1].y() - _v[0].y()) * (x - _v[0].x());
    bool e2 = (y - _v[1].y()) * (_v[2].x() - _v[1].x()) >= (_v[2].y() - _v[1].y()) * (x - _v[1].x());
    bool e3 = (y - _v[2].y()) * (_v[0].x() - _v[2].x()) >= (_v[0].y() - _v[2].y()) * (x - _v[2].x());
    return (e1 && e2 && e3) || (!(e1 || e2 || e3));
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Eigen::Vector4f* v) {
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return { c1,c2,c3 };
}

void Rasterizer::set_model(const Eigen::Matrix4f& m) 
{
    model = m;
}

void Rasterizer::set_view(const Eigen::Matrix4f& v) 
{
    view = v;
}

void Rasterizer::set_projection(const Eigen::Matrix4f& p) 
{
    projection = p;
}

void Rasterizer::set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader)
{
    vertex_shader = vert_shader;
}

void Rasterizer::set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader)
{
    fragment_shader = frag_shader;
}

Rasterizer::Rasterizer()
{
}

void Rasterizer::transform_triangle(int id, int thread_num, void * ppayload)
{
    RasterizerThreadPayload* payload = static_cast<RasterizerThreadPayload*>(ppayload);
    Rasterizer* r = payload->rasterizer;
    FrameBuffer* framebuffer = payload->framebuffer;

    Eigen::Matrix4f mvp = r->projection * r->view * r->model;

    for (int k = 0; k < payload->model_triangles->size(); k++)
    {
        if (k % thread_num != id) { continue; }
        const Triangle* t = &(payload->model_triangles->operator[](k));
        Triangle* newtri = &(payload->screen_triangles[k]);
        *newtri = *t;

        std::array<Eigen::Vector4f, 3> mm{
                (r->model * t->v[0]),
                (r->model * t->v[1]),
                (r->model * t->v[2])
        };

        std::transform(mm.begin(), mm.end(),
            payload->worldpos.begin() + 3 * k, [](auto& v) {
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
             
        Eigen::Matrix4f inv_trans = (r->model).inverse().transpose();
        Eigen::Vector4f n[] = {
                inv_trans * to_vec4(t->normal[0], 0.0f),
                inv_trans * to_vec4(t->normal[1], 0.0f),
                inv_trans * to_vec4(t->normal[2], 0.0f)
        };

        //Viewport transformation
        for (auto& vert : v)
        {
            vert.x() = 0.5 * framebuffer->width * (vert.x() + 1.0);
            vert.y() = 0.5 * framebuffer->height * (vert.y() + 1.0);
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

static Eigen::Vector3f lower_aabb(const Eigen::Vector4f* v) {
    int lower_x = std::max(static_cast<int>(
        std::floor(std::min(std::min(v[0].x(), v[1].x()), v[2].x()))), 0);
    int lower_y = std::max(static_cast<int>(
        std::floor(std::min(std::min(v[0].y(), v[1].y()), v[2].y()))), 0);
    float lower_z = std::min(std::min(v[0].z(), v[1].z()), v[2].z());
    return Eigen::Vector3f(lower_x, lower_y, lower_z);
}

static Eigen::Vector3f upper_aabb(const Eigen::Vector4f* v, int width, int height) {
    int upper_x = std::min(static_cast<int>(
        std::ceil(std::max(v[0].x(), std::max(v[1].x(), v[2].x())))), width);
    int upper_y = std::min(static_cast<int>(
        std::ceil(std::max(v[0].y(), std::max(v[1].y(), v[2].y())))), height);
    float upper_z = std::max(std::max(v[0].z(), v[1].z()), v[2].z());
    return Eigen::Vector3f(upper_x, upper_y, upper_z);
}

void Rasterizer::rasterize_triangle(int id, int thread_num, void * ppayload)
{
    RasterizerThreadPayload* payload = static_cast<RasterizerThreadPayload*>(ppayload);
    Rasterizer* r = payload->rasterizer;
    FrameBuffer* framebuffer = payload->framebuffer;
    
    for (int k = 0; k < payload->screen_triangles.size(); k++)
    {
        const Triangle* t = &(payload->screen_triangles[k]);
        std::array<Eigen::Vector3f, 3> world_pos = { payload->worldpos[k * 3],
                                                     payload->worldpos[k * 3 + 1],
                                                     payload->worldpos[k * 3 + 2] };
        auto v = t->toVector4();
        bool backface = (v[2].y() - v[0].y()) * (v[1].x() - v[0].x()) 
            <= (v[2].x() - v[0].x()) * (v[1].y() - v[0].y());

        if (!(r->backface_cull ^ backface)) { continue; }

        auto lower = lower_aabb(v.data());
        auto upper = upper_aabb(v.data(), framebuffer->width, framebuffer->height);
        if (upper.z() < -1 || lower.z() > 0) { continue; }

        for (int j = lower.y(); j < upper.y(); j++) {
            if (j % thread_num != id) { continue; }
            for (int i = lower.x(); i < upper.x(); i++) {
                if (!insideTriangle(i + 0.5, j + 0.5, t->v)) { continue; }
                int index = (framebuffer->height - j - 1) * framebuffer->width + i;

                auto [alpha, beta, gamma] = computeBarycentric2D(i + 0.5, j + 0.5, t->v);
                Eigen::Vector3f weight(alpha / v[0].w(), beta / v[1].w(), gamma / v[2].w());
                weight = weight / (weight.x() + weight.y() + weight.z());

                if (isnan(weight.x()) || isnan(weight.y()) || isnan(weight.z())) { continue; }
                if (isinf(weight.x()) || isinf(weight.y()) || isinf(weight.z())) { continue; }

                float depth = v[0].z() * alpha + v[1].z() * beta + v[2].z() * gamma;
                if (depth < -1 || depth > 0) { continue; }
                if (depth > framebuffer->depth_buf->operator[](index)) { continue; }
                framebuffer->depth_buf->operator[](index) = depth;

                if (framebuffer->frame_buf == nullptr) { continue; }

                Eigen::Vector3f color = interpolate(weight, t->color);
                Eigen::Vector3f normal = interpolate(weight, t->normal).normalized();
                Eigen::Vector2f coords = interpolate(weight, t->tex_coords);
                fragment_shader_payload fragment_payload(
                    color, normal, coords, &(r->uniform));

                fragment_payload.world_pos = interpolate(weight, world_pos.data());
                framebuffer->frame_buf->operator[](index) = r->fragment_shader(fragment_payload);
            }
        }
    }
}

void Rasterizer::draw(std::vector<Triangle>& TriangleList, JobThread & threads, FrameBuffer& framebuffer) {
    JobThread::ThreadTaskPayload task_payload;
    RasterizerThreadPayload data_payload;
    data_payload.rasterizer = this;
    data_payload.framebuffer = &(framebuffer);
    data_payload.model_triangles = &(TriangleList);
    data_payload.screen_triangles.resize(TriangleList.size());
    data_payload.worldpos.resize(TriangleList.size() * 3);
    task_payload.payload = static_cast<void *>(&data_payload);
    task_payload.task = transform_triangle;
    threads.wake_thread_job(task_payload);
    task_payload.task = rasterize_triangle;
    threads.wake_thread_job(task_payload);
}

//Screen space rasterization
