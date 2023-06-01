//
// Created by LEI XU on 4/27/19.
//

#pragma once
#include <vector>
#include <Eigen/Eigen>
#include "Texture.hpp"

class Camera;
class PointLight;

struct ShaderUniform
{
    Camera* camera;
    std::vector<PointLight*> lights;
    std::vector<Texture*> textures;
};

struct fragment_shader_payload
{
    fragment_shader_payload()
    {
        uniform = nullptr;
    }

    fragment_shader_payload(const Eigen::Vector3f& col, 
        const Eigen::Vector3f& nor,const Eigen::Vector2f& tc, ShaderUniform* unif) :
         color(col), normal(nor), tex_coords(tc), uniform(unif) {}


    Eigen::Vector3f world_pos;
    Eigen::Vector3f color;
    Eigen::Vector3f normal;
    Eigen::Vector2f tex_coords;
    ShaderUniform* uniform;
};

struct vertex_shader_payload
{
    Eigen::Vector3f position;
};


Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload);

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload);

Eigen::Vector3f phong_shadow_fragment_shader(const fragment_shader_payload& payload);

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload);

Eigen::Vector3f texture_shadow_fragment_shader(const fragment_shader_payload& payload);