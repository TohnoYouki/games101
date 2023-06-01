#include "Scene.hpp"
#include "OBJ_Loader.h"

auto vec3_to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Eigen::Vector4f(v3.x(), v3.y(), v3.z(), w);
}

std::vector<Triangle> plane(float width)
{
    std::vector<Triangle> result(2);
    Eigen::Vector4f vertices[4] = {
        {-width / 2, -width / 2, 0, 1.0}, {width / 2, -width / 2, 0, 1.0},
        {width / 2, width / 2, 0, 1.0}, {-width / 2, width / 2, 0, 1.0}
    };
    int indices[6] = { 0, 1, 3, 1, 2, 3 };
    for (int i = 0; i < 3; i++)
    {
        result[0].setVertex(i, vertices[indices[i]]);
        result[1].setVertex(i, vertices[indices[i + 3]]);
        result[0].setNormal(i, Eigen::Vector3f(0, 0, 1));
        result[1].setNormal(i, Eigen::Vector3f(0, 0, 1));
        result[0].setColor(i, 148, 121.0, 92.0);
        result[1].setColor(i, 148, 121.0, 92.0);
    }
    return result;
}

std::vector<Triangle> load_mesh(std::string path)
{
    objl::Loader Loader;
    bool loadout = Loader.LoadFile(path);
    std::vector<Triangle> result;

    for (auto mesh : Loader.LoadedMeshes)
    {
        for (int i = 0; i < mesh.Vertices.size(); i += 3)
        {
            Triangle t = Triangle();
            for (int j = 0; j < 3; j++)
            {
                t.setVertex(j, Eigen::Vector4f(
                    mesh.Vertices[i + j].Position.X,
                    mesh.Vertices[i + j].Position.Y,
                    mesh.Vertices[i + j].Position.Z, 1.0));
                t.setNormal(j, Eigen::Vector3f(
                    mesh.Vertices[i + j].Normal.X,
                    mesh.Vertices[i + j].Normal.Y,
                    mesh.Vertices[i + j].Normal.Z));
                t.setTexCoord(j, Eigen::Vector2f(
                    mesh.Vertices[i + j].TextureCoordinate.X,
                    mesh.Vertices[i + j].TextureCoordinate.Y));
            }
            result.push_back(t);
        }
    }
    return result;
}

static Eigen::Matrix4f rotation_matrix(Eigen::Vector3f angle)
{
    Eigen::Matrix4f x_mat;
    angle = angle * MY_PI / 180.0f;
    x_mat << cos(angle.x()), 0, sin(angle.x()), 0, 0, 1, 0, 0,
        -sin(angle.x()), 0, cos(angle.x()), 0, 0, 0, 0, 1;
    Eigen::Matrix4f y_mat;
    y_mat << 1, 0, 0, 0, 0, cos(angle.y()), -sin(angle.y()), 0,
        0, sin(angle.y()), cos(angle.y()), 0, 0, 0, 0, 1;
    Eigen::Matrix4f z_mat;
    z_mat << cos(angle.z()), -sin(angle.z()), 0, 0,
        sin(angle.z()), cos(angle.z()), 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1;
    return x_mat * y_mat * z_mat;
}

static Eigen::Matrix4f translate_matrix(Eigen::Vector3f position)
{
    Eigen::Matrix4f matrix;
    matrix << 1, 0, 0, position.x(), 0, 1, 0, position.y(),
        0, 0, 1, position.z(), 0, 0, 0, 1;
    return matrix;
}

PointLight::PointLight(Camera camera, Eigen::Vector3f intensity, int width, int height) :
    camera(camera), intensity(intensity), width(width), height(height)
{
    for (int i = 0; i < 6; i++) {
        shadowmaps[i].resize(width * height);
        Eigen::Vector3f front(i < 2, i >= 2 & i < 4, i >= 4);
        front = front * (-2 * (i % 2) + 1);
        Eigen::Vector3f up(0, i < 2 | i >= 4, i >= 2 & i < 4);
        up = up * (-2 * (i != 2) + 1);
        Eigen::Vector3f right = up.cross(front).normalized();
        up = front.cross(right).normalized();

        shadow_view_rotations[i] <<
            right.x(), right.y(), right.z(), 0,
            up.x(), up.y(), up.z(), 0,
            front.x(), front.y(), front.z(), 0,
            0, 0, 0, 1;
    }
}

Eigen::Matrix4f PointLight::shadow_view_matrix(int id)
{
    Eigen::Vector3f pos = position();
    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -pos.x(), 0, 1, 0, -pos.y(),
        0, 0, 1, -pos.z(), 0, 0, 0, 1;
    return shadow_view_rotations[id] * translate;
}

Transform::Transform(
    float scale, Eigen::Vector3f position, Eigen::Vector3f angle)
{
    Eigen::Matrix4f scale_mat;
    scale_mat << scale, 0, 0, 0, 0, scale, 0, 0,
                 0, 0, scale, 0, 0, 0, 0, 1;
    matrix = translate_matrix(position) * rotation_matrix(angle) * scale_mat;
}

void Transform::translate(Eigen::Vector3f position)
{
    position = matrix.block<3, 3>(0, 0)* position;
    matrix = translate_matrix(position) * matrix;
}

void Transform::rotate(Eigen::Vector3f angle)
{
    matrix = rotation_matrix(angle) * matrix;
}

Eigen::Matrix4f Camera::get_projection_matrix()
{
    Eigen::Matrix4f projection;
    double tanv = tan(eye_fov * MY_PI / 360.0f);
    projection << -1.0 / tanv, 0, 0, 0,
        0, -aspect_ratio / tanv, 0, 0,
        0, 0, zNear / (zFar - zNear), zNear * zFar / (zFar - zNear),
        0, 0, 1, 0;
    return projection;
}

static int cubemap_face(const Eigen::Vector3f& vector) {
    Eigen::Vector3f abs = vector.cwiseAbs();
    if (abs.x() > abs.y() && abs.x() > abs.z())
        return vector.x() < 0 ? 0 : 1;
    else if (abs.y() > abs.x() && abs.y() > abs.z())
        return vector.y() < 0 ? 2 : 3;
    else return vector.z() < 0 ? 4 : 5;
}

#define NUM_RINGS 3
#define NUM_SAMPLES 100
#define NUM_SEARCH 10
#define NUM_ARRAY 100

static void poissonDiskSamples(
    float seed, Eigen::Vector2f samples[], int number)
{
    float angle_step = 2 * MY_PI * float(NUM_RINGS) / float(number);
    float radius_step = 1.0 / float(number);
    float radius = radius_step, angle = seed * 2 * MY_PI;
    for (int i = 0; i < number; i++)
    {
        samples[i] = Eigen::Vector2f(std::cos(angle), std::sin(angle));
        samples[i] *= std::pow(radius, 0.75);
        radius += radius_step;
        angle += angle_step;
    }
}

void PointLight::generate_poisson_disk_samples() 
{
    pcf_samples.clear();
    pcf_samples.resize(NUM_ARRAY * NUM_SAMPLES);
    for (int i = 0; i < NUM_ARRAY; i++) {
        poissonDiskSamples((rand() % (10000) / 10000), 
            pcf_samples.data() + i * NUM_SAMPLES, NUM_SAMPLES);
    }
    search_samples.clear();
    search_samples.resize(NUM_ARRAY * NUM_SEARCH);
    for (int i = 0; i < NUM_ARRAY; i++) {
        poissonDiskSamples((rand() % (10000) / 10000),
            search_samples.data() + i * NUM_SEARCH, NUM_SEARCH);
    }
}

float PointLight::shadowmap(const Eigen::Vector3f& vector, float NdotL)
{
    const float bias = 5e-3;
    int id = cubemap_face(vector);
    Eigen::Matrix4f matrix = shadow_view_matrix(id);
    Eigen::Matrix4f project = camera.get_projection_matrix();
    matrix = project * matrix;
    Eigen::Vector4f target = matrix * vec3_to_vec4(vector + position(), 1.0);
    target /= target.w();
    float u = std::min(std::max((target.x() + 1.0f) * width * 0.5f, 0.0f), width - 0.1f);
    float v = std::min(std::max((target.y() + 1.0f) * height * 0.5f, 0.0f), height - 0.1f);
    int index = (height - int(v) - 1) * width + int(u);
    float depth = shadowmaps[id][index];
    if (isinf(depth)) { return 1.0; }
    depth = -project(2, 3) / (depth - project(2, 2));
    target.z() = -project(2, 3) / (target.z() - project(2, 2));
    return target.z() < depth + (2.0 - NdotL) * bias;
}

float PointLight::pcf(const Eigen::Vector3f& vector, float NdotL)
{
    float bias = (2.0 - NdotL) * 5e-3;
    const float world_radius = 0.2;
    int id = cubemap_face(vector);
    Eigen::Matrix4f matrix = shadow_view_matrix(id);
    Eigen::Matrix4f project = camera.get_projection_matrix();
    matrix = project * matrix;
    Eigen::Vector4f target = matrix * vec3_to_vec4(vector + position(), 1.0);
    target /= target.w();
    Eigen::Vector2f uv(
        std::min(std::max((target.x() + 1.0f) * 0.5f, 0.0f), 1 - 1e-5f),
        std::min(std::max((target.y() + 1.0f) * 0.5f, 0.0f), 1 - 1e-5f));
    float radius = world_radius / (2 * std::abs(vector(id / 2)));
    float depth = -project(2, 3) / (target.z() - project(2, 2));

    int samples = NUM_SAMPLES;
    int aindex = rand() % NUM_ARRAY;
    float shadow = 0.0;

    for (int i = 0; i < samples; i++)
    {
        Eigen::Vector2f offset = pcf_samples[aindex * NUM_SAMPLES + i];
        float u = std::min(std::max(uv.x() + offset.x() * radius, 0.0f), 1.0f - 1e-5f) * width;
        float v = std::min(std::max(uv.y() + offset.y() * radius, 0.0f), 1.0f - 1e-5f) * height;
        int index = (height - int(v) - 1) * width + int(u);
        float closest = shadowmaps[id][index];
        if (closest > 0) { shadow += 1.0; }
        closest = -project(2, 3) / (closest - project(2, 2));
        if (depth < closest + bias) { shadow += 1; }
    }
    return shadow / samples;
}

float PointLight::pcss(const Eigen::Vector3f& vector, float NdotL)
{
    float bias = (2.0 - NdotL) * 5e-3;
    int id = cubemap_face(vector);
    Eigen::Matrix4f matrix = shadow_view_matrix(id);
    Eigen::Matrix4f project = camera.get_projection_matrix();
    matrix = project * matrix;
    Eigen::Vector4f target = matrix * vec3_to_vec4(vector + position(), 1.0);
    target /= target.w();
    Eigen::Vector2f uv(
        std::min(std::max((target.x() + 1.0f) * 0.5f, 0.0f), 1 - 1e-5f),
        std::min(std::max((target.y() + 1.0f) * 0.5f, 0.0f), 1 - 1e-5f));
    float dreceiver = -project(2, 3) / (target.z() - project(2, 2));

    const float world_radius = 0.2;
    float radius = world_radius / (2 * std::abs(vector(id / 2)));
    int samples = NUM_SAMPLES;
    int aindex = rand() % NUM_ARRAY;

    float dblock = 0.0, count = 0.0;
    for (int i = 0; i < NUM_SEARCH; i++)
    {
        Eigen::Vector2f offset = search_samples[aindex * NUM_SEARCH + i];
        float u = std::min(std::max(uv.x() + offset.x() * radius, 0.0f), 1.0f - 1e-5f) * width;
        float v = std::min(std::max(uv.y() + offset.y() * radius, 0.0f), 1.0f - 1e-5f) * height;
        int index = (height - int(v) - 1) * width + int(u);
        float closest = shadowmaps[id][index];
        bool blocker = closest <= 0;
        closest = -project(2, 3) / (closest - project(2, 2));
        blocker &= dreceiver >= closest + bias;
        dblock += blocker ? closest : 0;
        count += blocker ? 1 : 0;
    }
    if (count == 0) { return 1.0; }
    dblock /= count;
    radius = (dreceiver - dblock) * lightsize / dblock;
    radius = radius / (2 * std::abs(vector(id / 2)));

    float shadow = 0.0;
    aindex = rand() % NUM_ARRAY;
    for (int i = 0; i < samples; i++)
    {
        Eigen::Vector2f offset = pcf_samples[aindex * NUM_SAMPLES + i];
        float u = std::min(std::max(uv.x() + offset.x() * radius, 0.0f), 1.0f - 1e-5f) * width;
        float v = std::min(std::max(uv.y() + offset.y() * radius, 0.0f), 1.0f - 1e-5f) * height;
        int index = (height - int(v) - 1) * width + int(u);
        float closest = shadowmaps[id][index];
        if (closest > 0) { shadow += 1.0; }
        closest = -project(2, 3) / (closest - project(2, 2));
        if (dreceiver < closest + bias) { shadow += 1; }
    }
    return shadow / samples;
}

float PointLight::shadow(const Eigen::Vector3f& vector, float NdotL)
{
    switch (shadow_type)
    {
    case 1: return pcf(vector, NdotL);
    case 2: return pcss(vector, NdotL);
    default: return shadowmap(vector, NdotL);
    }
}