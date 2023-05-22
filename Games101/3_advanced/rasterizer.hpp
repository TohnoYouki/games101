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

    enum class Primitive
    {
        Line,
        Triangle
    };

    /*
     * For the curious : The draw function takes two buffer id's as its arguments. These two structs
     * make sure that if you mix up with their orders, the compiler won't compile it.
     * Aka : Type safety
     * */
    struct pos_buf_id
    {
        int pos_id = 0;
    };

    struct ind_buf_id
    {
        int ind_id = 0;
    };

    struct col_buf_id
    {
        int col_id = 0;
    };

    struct ThreadPayload {
        int job_type = 0;
        int thread_num;
        bool exit = false;
        std::mutex mutex;
        int switch_id = 0;
        bool begin[2] = { false, false };
        int finished = 0;
        std::condition_variable cv;

        const std::vector<Triangle*>* orilist;
        std::vector<Triangle> triangles;
        std::vector<Vector3f> viewpos;
    };

    class rasterizer
    {
    public:
        static const int BUFNUM = 2;

        rasterizer(int w, int h, int thread_num_sqrt);

        void set_model(const Eigen::Matrix4f& m);
        void set_view(const Eigen::Matrix4f& v);
        void set_projection(const Eigen::Matrix4f& p);

        void set_texture(Texture tex) { texture = tex; }

        void set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader);
        void set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader);

        void clear(Buffers buff, int id);

        void draw(std::vector<Triangle *> &TriangleList, int id);

        std::vector<Eigen::Vector3f>& frame_buffer(int id) 
        { return frame_buf[id]; }

        ~rasterizer();
    private:

        void rasterize_triangle(int id);

        void transform_triangle(int id);

        void wake_thread_job();
        
        void thread_job(int id);

        // VERTEX SHADER -> MVP -> Clipping -> /.W -> VIEWPORT -> DRAWLINE/DRAWTRI -> FRAGSHADER

        
    private:
        Eigen::Matrix4f model;
        Eigen::Matrix4f view;
        Eigen::Matrix4f projection;

        int normal_id = -1;

        std::optional<Texture> texture;

        std::function<Eigen::Vector3f(fragment_shader_payload)> fragment_shader;
        std::function<Eigen::Vector3f(vertex_shader_payload)> vertex_shader;

        std::vector<std::thread> render_threads;
        std::vector<Eigen::Vector3f> frame_buf[BUFNUM];
        std::vector<float> depth_buf[BUFNUM];
        int get_index(int x, int y);

        int width, height;
        int draw_id;
        
        ThreadPayload thread_payload;
    };
}
