#include <iostream>
#include "Scene.hpp"
#include "global.hpp"
#include "Timer.hpp"
#include "Texture.hpp"
#include "Rasterizer.hpp"
#include <opencv2/opencv.hpp>

    int width = 700, height = 700;

    struct RenderThreadPayload
    {
        int key;
        bool exit = false;
        Scene scene;
        SwapChain swapchain = SwapChain(width, height);
        Rasterizer rasterizer = Rasterizer();
        JobThread threads = JobThread(8);
    };

    void build_scene(Scene & scene)
    {
        scene.meshes.push_back(load_mesh("C:/Users/ԶҰ��ϣ/Desktop/3_debug/models/spot/spot_triangulated_good.obj"));
        scene.meshes.push_back(plane(15));
        scene.textures.push_back(Texture("C:/Users/ԶҰ��ϣ/Desktop/3_debug/models/spot/spot_texture.png"));
        scene.camera = { Transform({1.0, {0, 9, 10}, {0.0, -45.0, 0.0}}), 45, 1, 0.1, 50 };
        scene.objects.push_back(MeshObject{ Transform{2.5, {2.5, 0.0, 0.0},
                                            {140.0, 0.0, 0.0}}, &(scene.meshes[0]),
                                             vertex_shader, texture_shadow_fragment_shader });
        scene.objects.push_back(MeshObject{ Transform{2.5, {-2.5, 0.0, 0.0},
                                            {-140.0, 0.0, 0.0}}, &(scene.meshes[0]),
                                             vertex_shader, texture_shadow_fragment_shader });
        scene.objects.push_back(MeshObject{ Transform{1.0, {0.0, -1.7, 0.0}, {0.0, -90.0, 0.0}},
                                            &(scene.meshes[1]), vertex_shader, phong_shadow_fragment_shader });
        scene.lights.push_back(PointLight{ Camera{ Transform{ 1.0, {10, 10, 10}, {0.0, 0.0, 0.0} },
                                           90, 1, 0.1, 50 }, { 100, 100, 100 }, width * 2, height * 2 });
        scene.lights.push_back(PointLight{ Camera { Transform{ 1.0, {-10, 10, 10}, {0.0, 0.0, 0.0} },
                                           90, 1, 0.1, 50 }, { 100, 100, 100 }, width * 2, height * 2 });
        scene.pcameras.push_back(&(scene.camera));
        for (auto& object : scene.lights) {
            scene.pcameras.push_back(&(object.camera));
        }
    }

    void update_scene(Scene & scene, int* key)
    {
        std::unique_lock<std::mutex> lock(scene.mutex);
        Transform* transform = nullptr;
        if (*key == 32) {
            scene.object_index = (scene.object_index + 1) % (scene.objects.size() + 1);
        }
        else if (*key == 13) {
            scene.object_index = 0;
            scene.camera_index = (scene.camera_index + 1) % scene.pcameras.size();
        }
        if (scene.object_index <= 0) {
            transform = &(scene.pcameras[scene.camera_index]->transform);
        }
        else { transform = &(scene.objects[scene.object_index - 1].transform); }

        switch (*key) {
        case 'w': transform->translate(Eigen::Vector3f(0, 0.1, 0)); break;
        case 's': transform->translate(Eigen::Vector3f(0, -0.1, 0)); break;
        case 'a': transform->translate(Eigen::Vector3f(-0.1, 0, 0)); break;
        case 'd': transform->translate(Eigen::Vector3f(0.1, 0, 0)); break;
        case 'q': transform->translate(Eigen::Vector3f(0, 0, 0.1)); break;
        case 'e': transform->translate(Eigen::Vector3f(0, 0, -0.1)); break;
        case 'j': transform->rotate(Eigen::Vector3f(-1.0, 0, 0)); break;
        case 'l': transform->rotate(Eigen::Vector3f(1.0, 0, 0)); break;
        case 'i': transform->rotate(Eigen::Vector3f(0, 1.0, 0)); break;
        case 'k': transform->rotate(Eigen::Vector3f(0, -1.0, 0)); break;
        case 'u': transform->rotate(Eigen::Vector3f(0, 0, -1.0)); break;
        case 'o':transform->rotate(Eigen::Vector3f(0, 0, 1.0)); break;
        }
    }

    void render_thread_fn(RenderThreadPayload * payload)
    {
        Rasterizer* r = &(payload->rasterizer);
        r->uniform.camera = &(payload->scene.camera);
        r->uniform.textures = { &(payload->scene.textures[0]) };
        r->uniform.lights = { &(payload->scene.lights[0]), &(payload->scene.lights[1]) };
        while (!payload->exit) {
            update_scene(payload->scene, &(payload->key));
            FrameBuffer buffer = payload->swapchain.get_next_render_framebuffer();
            buffer.clear(BufferTypes::Color | BufferTypes::Depth);
            
            r->backface_cull = false;
            for (auto& light : payload->scene.lights) {
                for (int i = 0; i < 6; i++) {
                    FrameBuffer depth_buf = FrameBuffer{ light.width, light.height, &(light.shadowmaps[i]), nullptr };
                    depth_buf.clear(BufferTypes::Depth);
                    for (int j = 0; j < payload->scene.objects.size(); j++) {
                        r->set_vertex_shader(payload->scene.objects[j].vertex_shader);
                        r->set_fragment_shader(nullptr);
                        r->set_model(payload->scene.objects[j].transform.get_model_matrix());
                        r->set_view(light.shadow_view_matrix(i));
                        r->set_projection(light.camera.get_projection_matrix());
                        r->draw(*(payload->scene.objects[j].mesh), payload->threads, depth_buf);
                    }
                }
            }

            r->backface_cull = true;
            Camera* camera = payload->scene.pcameras[payload->scene.camera_index];
            for (auto& light : payload->scene.lights) {
                light.generate_poisson_disk_samples();
            }
            for (int i = 0; i < payload->scene.objects.size(); i++)
            {
                r->set_vertex_shader(payload->scene.objects[i].vertex_shader);
                r->set_fragment_shader(payload->scene.objects[i].fragment_shader);
                r->set_model(payload->scene.objects[i].transform.get_model_matrix());
                r->set_view(camera->get_view_matrix());
                r->set_projection(camera->get_projection_matrix());
                r->draw(*(payload->scene.objects[i].mesh), payload->threads, buffer);
            }
        }
    }

    int main(int argc, const char** argv)
    {
        RenderThreadPayload payload;
        build_scene(payload.scene);
        std::thread render_thread(render_thread_fn, &payload);

        FrameCounter frame_counter(0.5);
        while (payload.key != 27)
        {
            float fps = frame_counter.latest_fps();
            FrameBuffer buffer = payload.swapchain.get_next_present_framebuffer();
            cv::Mat image(buffer.width, buffer.height, CV_32FC3, buffer.frame_buf->data());
            cv::putText(image, "fps:" + std::to_string(fps), cv::Point(50, 50),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255));
            image.convertTo(image, CV_8UC3, 1.0f);
            cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
            cv::imshow("image", image);

            std::unique_lock<std::mutex> lock(payload.scene.mutex);
            payload.key = cv::waitKey(1);
            lock.unlock();
        }

        payload.exit = true;
        render_thread.join();
        return 0;
    }