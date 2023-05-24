//
// Created by goksu on 4/6/19.
//

#pragma once

#include <Eigen/Eigen>
#include <thread>
#include <atomic>
#include <optional>
#include <algorithm>
#include "global.hpp"
#include "Shader.hpp"
#include "Triangle.hpp"

using namespace Eigen;

namespace rst
{
    enum class Buffers
    {
        Color = 1,
        Depth = 2
    };

    inline Buffers operator|(Buffers a, Buffers b)
    {
        return Buffers((int)a | (int)b);
    }

    inline Buffers operator&(Buffers a, Buffers b)
    {
        return Buffers((int)a & (int)b);
    }

    class JobThread
    {
    public:
        typedef void (*ThreadTask) (int, int, void *);

        struct ThreadTaskPayload {
            ThreadTask task;
            void* payload;
        };
        struct ThreadControlPayload {
            int thread_num;
            bool exit = false;
            std::mutex mutex;

            int wake_semaphore_index = 0;
            bool wake_semaphore[2] = { false, false };
            int finished_thread = 0;
            std::condition_variable cv;
        };

        JobThread(int num);
        ~JobThread();

        void wake_thread_job(const ThreadTaskPayload& payload);
        void thread_job(int id);
        
    private:
        std::vector<std::thread> threads;
        ThreadControlPayload control_payload;
        const ThreadTaskPayload* task_payload;
    };

    class Rasterizer
    {
    public:
        static const int SWAPCHAIN_NUM = 2;

        struct RasterizerThreadPayload {
            Rasterizer* rasterizer;
            int draw_id;
            const std::vector<Triangle*>* model_triangles;
            std::vector<Vector3f> viewpos;
            std::vector<Triangle> screen_triangles;
        };
    private:
        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        std::optional<Texture> texture;

        std::function<Eigen::Vector3f(vertex_shader_payload)> vertex_shader;
        std::function<Eigen::Vector3f(fragment_shader_payload)> fragment_shader;

        int width, height;
        std::vector<Eigen::Vector3f> frame_buf[SWAPCHAIN_NUM];
        std::vector<float> depth_buf;
    public:
        Rasterizer(int w, int h);

        void set_model(const Eigen::Matrix4f& m);
        void set_view(const Eigen::Matrix4f& v);
        void set_projection(const Eigen::Matrix4f& p);

        void set_texture(Texture tex) { texture = tex; }

        void set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader);
        void set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader);

        void clear(Buffers buff, int id);
        void draw(std::vector<Triangle *> &TriangleList, int id, JobThread& threads);

        std::vector<Eigen::Vector3f>& frame_buffer(int id) { return frame_buf[id]; }

    private:
        static void rasterize_triangle(int id, int thread_num, void * ppayload);
        static void transform_triangle(int id, int thread_num, void * ppayload);
    };
}