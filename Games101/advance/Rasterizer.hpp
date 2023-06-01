//
// Created by goksu on 4/6/19.
//

#pragma once

#include <Eigen/Eigen>
#include <optional>
#include <algorithm>
#include "global.hpp"
#include "Shader.hpp"
#include "Triangle.hpp"
#include "SwapChain.hpp"
#include "JobThread.hpp"

class Rasterizer
{
public:
    bool backface_cull = true;
    struct RasterizerThreadPayload {
        std::mutex mutex;
        Rasterizer* rasterizer;
        FrameBuffer* framebuffer;
        const std::vector<Triangle>* model_triangles;
        std::vector<Eigen::Vector3f> worldpos;
        std::vector<Triangle> screen_triangles;
    };
private:
    Eigen::Matrix4f model;
    Eigen::Matrix4f view;
    Eigen::Matrix4f projection;
    std::function<Eigen::Vector3f(vertex_shader_payload)> vertex_shader;
    std::function<Eigen::Vector3f(fragment_shader_payload)> fragment_shader;
public:
    Rasterizer();

    ShaderUniform uniform;
    void set_model(const Eigen::Matrix4f& m);
    void set_view(const Eigen::Matrix4f& v);
    void set_projection(const Eigen::Matrix4f& p);

    void set_vertex_shader(std::function<Eigen::Vector3f(vertex_shader_payload)> vert_shader);
    void set_fragment_shader(std::function<Eigen::Vector3f(fragment_shader_payload)> frag_shader);

    void draw(std::vector<Triangle>& TriangleList,
        JobThread& threads, FrameBuffer& framebuffer);

private:
    static int clip(const Eigen::Vector4f v[], Eigen::Vector3f result[]);
    static void rasterize_triangle(int id, int thread_num, void* ppayload);
    static void transform_triangle(int id, int thread_num, void* ppayload);
};