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

static int cubemap_face(const Eigen::Vector3f vector) {
    Eigen::Vector3f abs = vector.cwiseAbs();
    if (abs.x() > abs.y() && abs.x() > abs.z())
        return vector.x() < 0 ? 0 : 1;
    else if (abs.y() > abs.x() && abs.y() > abs.z())
        return vector.y() < 0 ? 2 : 3;
    else return vector.z() < 0 ? 4 : 5;
}

float PointLight::sample_shadowmap(Eigen::Vector3f vector, float NdotL) {
    int id = cubemap_face(vector);
    Eigen::Matrix4f matrix = shadow_view_matrix(id);
    Eigen::Matrix4f project = camera.get_projection_matrix();
    matrix = project * matrix;
    Eigen::Vector4f target = matrix * vec3_to_vec4(vector + position(), 1.0);
    target /= target.w();
    float u = std::min(std::max((target.x() + 1.0f) * width * 0.5f, 0.0f), width - 0.1f);
    float v = std::min(std::max((target.y() + 1.0f) * height * 0.5f, 0.0f), height - 0.1f);
    int index = (height - int(v) - 1) * width + int(u);
    float depth = -project(11) / (shadowmaps[id][index] - project(10));
    target.z() = -project(11) / (target.z() - project(10));
    return target.z() < depth + 5e-1 + (1.0 - NdotL) * 5e-1;
}

float PointLight::pcss(Eigen::Vector3f vector, float NdotL)
{
    int samples = 21;
    static Eigen::Vector3f sampleOffsetDirections[21] =
    {
        Eigen::Vector3f(0, 0, 0),
        Eigen::Vector3f(1, 1, 1), Eigen::Vector3f(1, -1, 1),
        Eigen::Vector3f(-1, -1, 1), Eigen::Vector3f(-1, 1, 1),
        Eigen::Vector3f(1, 1, -1), Eigen::Vector3f(1, -1, -1),
        Eigen::Vector3f(-1, -1, -1), Eigen::Vector3f(-1, 1, -1),
        Eigen::Vector3f(1, 1, 0), Eigen::Vector3f(1, -1, 0),
        Eigen::Vector3f(-1, -1, 0), Eigen::Vector3f(-1, 1, 0),
        Eigen::Vector3f(1, 0, 1), Eigen::Vector3f(-1, 0, 1),
        Eigen::Vector3f(1, 0, -1), Eigen::Vector3f(-1, 0, -1),
        Eigen::Vector3f(0, 1, 1), Eigen::Vector3f(0, -1, 1),
        Eigen::Vector3f(0, -1, -1), Eigen::Vector3f(0, 1, -1)
    };
    Eigen::Matrix4f project = camera.get_projection_matrix();

    float dblock = 0.0, dreceiver = 0.0;
    float diskRadius = 0.1;
    for (int i = 0; i < samples; i++) {
        Eigen::Vector3f sample = vector + sampleOffsetDirections[i] * diskRadius;
        int id = cubemap_face(sample);
        Eigen::Matrix4f matrix = shadow_view_matrix(id);
        matrix = project * matrix;
        Eigen::Vector4f target = matrix * vec3_to_vec4(sample + position(), 1.0);
        target /= target.w();
        float u = std::min(std::max((target.x() + 1.0f) * width * 0.5f, 0.0f), width - 0.1f);
        float v = std::min(std::max((target.y() + 1.0f) * height * 0.5f, 0.0f), height - 0.1f);
        int index = (height - int(v) - 1) * width + int(u);
        if (i == 0) { dreceiver = -project(11) / (target.z() - project(10)); }
        dblock += -project(11) / (shadowmaps[id][index] - project(10));
    }
    dblock /= samples;
    float radius = (dreceiver - dblock) * lightsize / dblock;
    if (radius <= 1e-5) { return 1.0; }

    float shadow = 0.0;
    for (int i = 0; i < samples; i++) {
        Eigen::Vector3f sample = vector + sampleOffsetDirections[i] * radius;
        int id = cubemap_face(sample);
        Eigen::Matrix4f matrix = shadow_view_matrix(id);
        matrix = project * matrix;
        Eigen::Vector4f target = matrix * vec3_to_vec4(sample + position(), 1.0);
        target /= target.w();
        float u = std::min(std::max((target.x() + 1.0f) * width * 0.5f, 0.0f), width - 0.1f);
        float v = std::min(std::max((target.y() + 1.0f) * height * 0.5f, 0.0f), height - 0.1f);
        int index = (height - int(v) - 1) * width + int(u);
        float depth = -project(11) / (shadowmaps[id][index] - project(10));
        if (dreceiver < depth + 1 + (1.0 - NdotL) * 1) { shadow += 1; }
    }
    return shadow / samples;
}