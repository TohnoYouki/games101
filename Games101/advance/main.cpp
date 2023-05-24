#include <iostream>
#include <opencv2/opencv.hpp>

#include <thread>
#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle, Vector3f position)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, position.x(),
            0, 1, 0, position.y(),
            0, 0, 1, position.z(),
            0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Use the same projection matrix from the previous assignments
    Eigen::Matrix4f projection;
    double tanv = tan(eye_fov / 2);
    projection << -1.0 / tanv, 0, 0, 0,
        0, -aspect_ratio / tanv, 0, 0,
        0, 0, (zNear + zFar) / (zNear - zFar), 2 * zNear * zFar / (zFar - zNear),
        0, 0, 1, 0;
    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = { 0, 0, 0 };
    if (payload.texture)
    {
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{ {20, 20, 20}, {500, 500, 500} };
    auto l2 = light{ {-20, 20, 0}, {500, 500, 500} };

    std::vector<light> lights = { l1, l2 };
    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
    Eigen::Vector3f eye_pos{ 0, 0, 10 };

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = { 0, 0, 0 };

    Eigen::Vector3f ambient = amb_light_intensity;
    Eigen::Vector3f diffuse = { 0, 0, 0 };
    Eigen::Vector3f specular = { 0, 0, 0 };

    Vector3f output_vector = (eye_pos - point).normalized();

    for (auto& light : lights)
    {
        Vector3f light_vector = light.position - point;
        float light_attenuation = light_vector.squaredNorm();
        light_vector = light_vector.normalized();
        Vector3f half_vector = (light_vector + output_vector).normalized();

        diffuse += light.intensity * std::max(normal.dot(light_vector), 0.0f) / light_attenuation;
        specular += light.intensity * std::pow(std::max(normal.dot(half_vector), 0.0f), p) / light_attenuation;
    }

    result_color = ka.cwiseProduct(ambient) + kd.cwiseProduct(diffuse) + ks.cwiseProduct(specular);

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    Eigen::Vector3f ambient = amb_light_intensity;
    Eigen::Vector3f diffuse = { 0, 0, 0 };
    Eigen::Vector3f specular = { 0, 0, 0 };

    Vector3f output_vector = (eye_pos - point).normalized();

    for (auto& light : lights)
    {
        Vector3f light_vector = light.position - point;
        float light_attenuation = light_vector.squaredNorm();
        light_vector = light_vector.normalized();
        Vector3f half_vector = (light_vector + output_vector).normalized();

        diffuse += light.intensity * std::max(normal.dot(light_vector), 0.0f) / light_attenuation;
        specular += light.intensity * std::pow(std::max(normal.dot(half_vector), 0.0f), p) / light_attenuation;
    }

    result_color = ka.cwiseProduct(ambient) + kd.cwiseProduct(diffuse) + ks.cwiseProduct(specular);
    return result_color * 255.f;
}

struct RasterizerPayload
{
    Vector3f eye_pos;
    float angle;
    float position;
    std::vector<Triangle*> triangleList;
};

struct RenderThreadPayload 
{
    bool exit = false;
    std::mutex framebuf_mutexs[rst::Rasterizer::SWAPCHAIN_NUM];
    rst::Rasterizer rasterizer = rst::Rasterizer(700, 700);
    rst::JobThread threads = rst::JobThread(8);
    std::mutex rasterizer_mutex;
    RasterizerPayload rasterizer_payload;
};

void render_thread_fn(RenderThreadPayload * payload)
{
    int image_index = 0;
    while (!payload->exit) {
        std::unique_lock<std::mutex> lock(payload->framebuf_mutexs[image_index]);

        std::unique_lock<std::mutex> data_lock(payload->rasterizer_mutex);
        float angle = payload->rasterizer_payload.angle;
        Vector3f eye_pos = payload->rasterizer_payload.eye_pos;
        float position = payload->rasterizer_payload.position;
        data_lock.unlock();

        rst::Rasterizer* r = &(payload->rasterizer);
        r->clear(rst::Buffers::Color | rst::Buffers::Depth, image_index);

        r->set_model(get_model_matrix(angle, Vector3f(position, 0.0f, 0.0f)));
        r->set_view(get_view_matrix(eye_pos));
        r->set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
        r->draw(payload->rasterizer_payload.triangleList, image_index, payload->threads);

        r->set_model(get_model_matrix(-angle, Vector3f(-position, 0.0f, 0.0f)));
        r->set_view(get_view_matrix(eye_pos));
        r->set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
        r->draw(payload->rasterizer_payload.triangleList, image_index, payload->threads);

        image_index = (image_index + 1) % payload->rasterizer.SWAPCHAIN_NUM;
        //data_lock.unlock();
        lock.unlock();
    }
}

int main(int argc, const char** argv)
{
    RenderThreadPayload payload;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../models/spot/";
    // Load .obj File
    bool loadout = Loader.LoadFile("../models/spot/spot_triangulated_good.obj");
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,
                                        mesh.Vertices[i+j].Position.Y,
                                        mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,
                                        mesh.Vertices[i+j].Normal.Y,
                                        mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, 
                                          mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            payload.rasterizer_payload.triangleList.push_back(t);
        }
    }

    payload.rasterizer_payload.eye_pos = { 0,0,10 };
    payload.rasterizer_payload.angle = 140.0;
    payload.rasterizer_payload.position = 2.5f;
    
    auto texture_path = "spot_texture.png";
    payload.rasterizer.set_texture(Texture(obj_path + texture_path));

    payload.rasterizer.set_vertex_shader(vertex_shader);
    payload.rasterizer.set_fragment_shader(texture_fragment_shader);

    std::thread render_thread(render_thread_fn, &payload);

    float fps = 0.0;
    int key = 0, image_index = 0;
    int frame_count = 0, last_frame_count = 0;
    clock_t last_time = clock(), now_time = clock();
    
    while(key != 27)
    {
        now_time = clock();
        if (double(now_time - last_time) >= CLOCKS_PER_SEC * 0.5) {
            fps = (frame_count - last_frame_count) * CLOCKS_PER_SEC / double(now_time - last_time);
            last_time = now_time;
            last_frame_count = frame_count;
        }

        std::unique_lock<std::mutex> lock(payload.framebuf_mutexs[image_index]);

        cv::Mat image(700, 700, CV_32FC3, payload.rasterizer.frame_buffer(image_index).data());
        cv::putText(image, "fps:" + std::to_string(fps), cv::Point(50, 50),
            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255));
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);

        lock.unlock();

        key = cv::waitKey(1);
        std::unique_lock<std::mutex> data_lock(payload.rasterizer_mutex);
        if (key == 'a' ) {
            payload.rasterizer_payload.angle -= 1;
        }
        else if (key == 'd') {
            payload.rasterizer_payload.angle += 1;
        }
        else if (key == 'w') {
            payload.rasterizer_payload.position += 0.1;
        }
        else if (key == 's') {
            payload.rasterizer_payload.position -= 0.1;
        }
        data_lock.unlock();

        image_index = (image_index + 1) % payload.rasterizer.SWAPCHAIN_NUM;
        frame_count++;
    }
    payload.exit = true;
    render_thread.join();
    return 0;
}
