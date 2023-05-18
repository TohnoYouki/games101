//
// Created by goksu on 2/25/20.
//

#include <fstream>
#include <thread>
#include <atomic>
#include "Scene.hpp"
#include "Renderer.hpp"


inline float deg2rad(const float& deg) { return deg * M_PI / 180.0; }

const float EPSILON = 0.0001;

void Renderer::BlockRender(const Scene* scene,
    std::vector<Vector3f> * framebuffer, std::atomic<int>* id)
{
    float scale = tan(deg2rad(scene->fov * 0.5));
    float imageAspectRatio = scene->width / (float)scene->height;

    while (true) {
        int m = id->fetch_add(1);
        if (m >= scene->width * scene->height) { break; }

        int j = m / scene->width;
        int i = m % (scene->width);
        float x = (2 * (i + 0.5) / (float)scene->width - 1) *
            imageAspectRatio * scale;
        float y = (1 - 2 * (j + 0.5) / (float)scene->height) * scale;

        Vector3f dir = normalize(Vector3f(-x, y, 1));
        for (int k = 0; k < spp; k++) {
            framebuffer->operator[](m) += scene->castRay(Ray(eye_pos, dir), 0) / spp;
        }
    }
}

// The main render function. This where we iterate over all pixels in the image,
// generate primary rays and cast these rays into the scene. The content of the
// framebuffer is saved to a file.
void Renderer::Render(const Scene& scene)
{
    spp = 256;
    eye_pos = Vector3f(278, 273, -800);
    std::vector<Vector3f> framebuffer(scene.width * scene.height);
    std::cout << "SPP: " << spp << "\n";

    int thread_num = 16;
    std::atomic<int> id = 0;
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_num; i++) {
        threads.emplace_back(&Renderer::BlockRender, this, &scene, &framebuffer, &id);
        threads[i].detach();
    }

    while (id.load() < scene.width * scene.height + thread_num) {
        UpdateProgress(id.load() / (float)(scene.height * scene.width));
        _sleep(20);
    }

    UpdateProgress(1.f);
    // save framebuffer to file
    FILE* fp = fopen("binary.ppm", "wb");
    (void)fprintf(fp, "P6\n%d %d\n255\n", scene.width, scene.height);
    for (auto i = 0; i < scene.height * scene.width; ++i) {
        static unsigned char color[3];
        color[0] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].x), 0.6f));
        color[1] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].y), 0.6f));
        color[2] = (unsigned char)(255 * std::pow(clamp(0, 1, framebuffer[i].z), 0.6f));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);    
}
