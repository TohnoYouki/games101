//
// Created by goksu on 2/25/20.
//
#include "Scene.hpp"

#pragma once
struct hit_payload
{
    float tNear;
    uint32_t index;
    Vector2f uv;
    Object* hit_obj;
};

class Renderer
{
public:
    void BlockRender(const Scene* scene,
        std::vector<Vector3f>* framebuffer, std::atomic<int>* id);

    void Render(const Scene& scene);

private:

    int spp;

    Vector3f eye_pos;
};
