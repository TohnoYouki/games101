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

static Eigen::Vector4f interpolate(const Eigen::Vector3f& weight, const Eigen::Vector4f* attribute)
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

int Rasterizer::clip(const Eigen::Vector4f v[], Eigen::Vector3f result[])
{
    int count[2] = { 1,0 };
    Eigen::Vector2f weights[2][45] = { {{1, 0}, {0, 1}, {0, 0}}, {} };
    Eigen::Vector4f positions[2][45] = { {v[0], v[1], v[2]}, {} };

    auto ratio = [](const Eigen::Vector4f vertex, int type) {
        return type == 0 ? vertex.w() : (type == 6 ? -vertex.z() :
            vertex[(type - 1) / 2] * (-2 * ((type - 1) % 2) + 1) + vertex.w());
    };

    for (int i = 0; i < 7; i++) {
        count[(i + 1) % 2] = 0;
        for (int j = 0; j < count[i % 2]; j++)
        {
            float r[3] = { ratio(positions[i % 2][j * 3], i),
                           ratio(positions[i % 2][j * 3 + 1], i),
                           ratio(positions[i % 2][j * 3 + 2], i) };
            float t[3] = { r[0] / (r[0] - r[1]), r[1] / (r[1] - r[2]), r[2] / (r[2] - r[0]) };
            bool inside[3] = { r[0] <= 0, r[1] <= 0, r[2] <= 0 };

            float factors[4];
            int vcount = 0;
            for (int k = 0; k < 3; k++) {
                if (inside[k]) { factors[vcount++] = k; }
                if (inside[k] != inside[(k + 1) % 3]) { 
                    if (t[k] > 0 && t[k] < 1)
                        factors[vcount++] = t[k] + k;
                }
            }

            for (int k = 0; k + 2 <= vcount; k += 2) {
                for (int h = 0; h < 3; h++) {
                    int prev = int(factors[(k + h) % vcount]);
                    int next = (prev + 1) % 3;
                    float ratio = factors[(k + h) % vcount] - prev;
                    positions[(i + 1) % 2][count[(i + 1) % 2] * 3 + h] =
                        positions[i % 2][j * 3 + prev] * (1 - ratio) +
                        positions[i % 2][j * 3 + next] * ratio;
                    weights[(i + 1) % 2][count[(i + 1) % 2] * 3 + h] =
                        weights[i % 2][j * 3 + prev] * (1 - ratio) +
                        weights[i % 2][j * 3 + next] * ratio;
                }
                count[(i + 1) % 2]++;
            }
        }
    }
    for (int i = 0; i < count[1]; i++) {
        for (int j = 0; j < 3; j++) {
            result[i * 3 + j].x() = weights[1][i * 3 + j].x();
            result[i * 3 + j].y() = weights[1][i * 3 + j].y();
            result[i * 3 + j].z() = 1 - result[i * 3 + j].x() - result[i * 3 + j].y();
        }
    }
    return count[1];
}

void Rasterizer::transform_triangle(int id, int thread_num, void* ppayload)
{
    RasterizerThreadPayload* payload = static_cast<RasterizerThreadPayload*>(ppayload);
    Rasterizer* r = payload->rasterizer;
    FrameBuffer* framebuffer = payload->framebuffer;

    Eigen::Vector3f weights[45];
    Eigen::Matrix4f mvp = r->projection * r->view * r->model;
    Eigen::Matrix4f inv_trans = (r->model).inverse().transpose();

    for (int k = 0; k < payload->model_triangles->size(); k++)
    {
        if (k % thread_num != id) { continue; }
        const Triangle* t = &(payload->model_triangles->operator[](k));

        std::array<Eigen::Vector4f, 3> mm{
                (r->model * t->v[0]),
                (r->model * t->v[1]),
                (r->model * t->v[2])
        };
        Eigen::Vector4f v[] = {
                mvp * t->v[0],
                mvp * t->v[1],
                mvp * t->v[2]
        };

        int count = Rasterizer::clip(v, weights);
        for (int i = 0; i < count; i++)
        {
            Triangle newtri;
            Eigen::Vector3f worldpos[3];
            for (int j = 0; j < 3; j++)
            {
                newtri.v[j] = interpolate(weights[i * 3 + j], v);
                newtri.normal[j] = interpolate(weights[i * 3 + j], t->normal);
                newtri.color[j] = interpolate(weights[i * 3 + j], t->color);
                newtri.tex_coords[j] = interpolate(weights[i * 3 + j], t->tex_coords);
                newtri.tex = t->tex;
                worldpos[j] = interpolate(weights[i * 3 + j], mm.data()).head<3>();
            }

            for (auto& vec : newtri.v) {
                vec.x() /= vec.w();
                vec.y() /= vec.w();
                vec.z() /= vec.w();
                vec.x() = 0.5 * framebuffer->width * (vec.x() + 1.0);
                vec.y() = 0.5 * framebuffer->height * (vec.y() + 1.0);
            }
            for (auto& n : newtri.normal) {
                n = (inv_trans * to_vec4(n, 0.0f)).head<3>();
            }
            newtri.color[0] = { 148 / 255.0, 121.0 / 255.0, 92.0 / 255.0 };
            newtri.color[1] = { 148 / 255.0, 121.0 / 255.0, 92.0 / 255.0 };
            newtri.color[2] = { 148 / 255.0, 121.0 / 255.0, 92.0 / 255.0 };

            std::unique_lock<std::mutex> lock(payload->mutex);
            payload->screen_triangles.push_back(newtri);
            for (int j = 0; j < 3; j++)
                payload->worldpos.push_back(worldpos[j]);
            lock.unlock();
        }
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

void Rasterizer::rasterize_triangle(int id, int thread_num, void* ppayload)
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
            < (v[2].x() - v[0].x()) * (v[1].y() - v[0].y());
        if (!(r->backface_cull ^ backface)) { continue; }

        auto lower = lower_aabb(v.data());
        auto upper = upper_aabb(v.data(), framebuffer->width, framebuffer->height);

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

void Rasterizer::draw(std::vector<Triangle>& TriangleList, JobThread& threads, FrameBuffer& framebuffer) {
    JobThread::ThreadTaskPayload task_payload;
    RasterizerThreadPayload data_payload;
    data_payload.rasterizer = this;
    data_payload.framebuffer = &(framebuffer);
    data_payload.model_triangles = &(TriangleList);
    data_payload.screen_triangles.reserve(TriangleList.size());
    data_payload.worldpos.reserve(TriangleList.size() * 3);
    task_payload.payload = static_cast<void*>(&data_payload);
    task_payload.task = transform_triangle;
    threads.wake_thread_job(task_payload);
    task_payload.task = rasterize_triangle;
    threads.wake_thread_job(task_payload);
}

//Screen space rasterization
