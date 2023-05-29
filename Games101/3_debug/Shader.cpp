#include "Shader.hpp"
#include "Scene.hpp"

Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
    Eigen::Vector3f eye_pos = payload.uniform->camera->position();

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.world_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = { 0, 0, 0 };
    Eigen::Vector3f ambient = amb_light_intensity;
    Eigen::Vector3f diffuse = { 0, 0, 0 };
    Eigen::Vector3f specular = { 0, 0, 0 };

    Eigen::Vector3f output_vector = (eye_pos - point).normalized();

    for (auto light : payload.uniform->lights)
    {
        Eigen::Vector3f light_vector = light->position() - point;
        float light_attenuation = light_vector.squaredNorm();
        light_vector = light_vector.normalized();
        Eigen::Vector3f half_vector = (light_vector + output_vector).normalized();

        diffuse += light->intensity * std::max(normal.dot(light_vector), 0.0f) / light_attenuation;
        specular += light->intensity * std::pow(std::max(normal.dot(half_vector), 0.0f), p) / light_attenuation;
    }

    result_color = ka.cwiseProduct(ambient) + kd.cwiseProduct(diffuse) + ks.cwiseProduct(specular);
    return result_color * 255.f;
}

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = { 0, 0, 0 };
    if (payload.uniform->textures.size() > 0)
    {
        return_color = payload.uniform->textures[0]->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
    Eigen::Vector3f eye_pos = payload.uniform->camera->position();

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.world_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = { 0, 0, 0 };

    Eigen::Vector3f ambient = amb_light_intensity;
    Eigen::Vector3f diffuse = { 0, 0, 0 };
    Eigen::Vector3f specular = { 0, 0, 0 };

    Eigen::Vector3f output_vector = (eye_pos - point).normalized();

    for (auto light : payload.uniform->lights)
    {
        Eigen::Vector3f light_vector = light->position() - point;
        float light_attenuation = light_vector.squaredNorm();
        light_vector = light_vector.normalized();
        Eigen::Vector3f half_vector = (light_vector + output_vector).normalized();

        diffuse += light->intensity * std::max(normal.dot(light_vector), 0.0f) / light_attenuation;
        specular += light->intensity * std::pow(std::max(normal.dot(half_vector), 0.0f), p) / light_attenuation;
    }

    result_color = ka.cwiseProduct(ambient) + kd.cwiseProduct(diffuse) + ks.cwiseProduct(specular);

    return result_color * 255.f;
}

Eigen::Vector3f phong_shadow_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
    Eigen::Vector3f eye_pos = payload.uniform->camera->position();

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.world_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = { 0, 0, 0 };
    Eigen::Vector3f ambient = amb_light_intensity;
    Eigen::Vector3f diffuse = { 0, 0, 0 };
    Eigen::Vector3f specular = { 0, 0, 0 };

    Eigen::Vector3f out_v = (eye_pos - point).normalized();
    for (auto light : payload.uniform->lights)
    {  
        Eigen::Vector3f light_v = light->position() - point;
        float light_attenuation = 1.0 / light_v.squaredNorm();
        light_v = light_v.normalized();
        Eigen::Vector3f half_v = (light_v + out_v).normalized();
        float NdotL = normal.dot(light_v);
        float shadow = light->sample_shadowmap(point - light->position(), NdotL);
        diffuse += light->intensity * shadow * std::max(NdotL, 0.0f) * light_attenuation;
        specular += light->intensity * shadow * std::pow(std::max(normal.dot(half_v), 0.0f), p) * light_attenuation;
    }

    result_color = ka.cwiseProduct(ambient) + kd.cwiseProduct(diffuse) + ks.cwiseProduct(specular);
    return result_color * 255.f;
}

Eigen::Vector3f texture_shadow_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = { 0, 0, 0 };
    if (payload.uniform->textures.size() > 0)
    {
        return_color = payload.uniform->textures[0]->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    Eigen::Vector3f amb_light_intensity{ 10, 10, 10 };
    Eigen::Vector3f eye_pos = payload.uniform->camera->position();

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.world_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = { 0, 0, 0 };

    Eigen::Vector3f ambient = amb_light_intensity;
    Eigen::Vector3f diffuse = { 0, 0, 0 };
    Eigen::Vector3f specular = { 0, 0, 0 };

    Eigen::Vector3f out_v = (eye_pos - point).normalized();
    for (auto light : payload.uniform->lights)
    {
        Eigen::Vector3f light_v = light->position() - point;
        float light_attenuation = 1.0 / light_v.squaredNorm();
        light_v = light_v.normalized();
        Eigen::Vector3f half_v = (light_v + out_v).normalized();
        float NdotL = normal.dot(light_v);
        float shadow = light->sample_shadowmap(point - light->position(), NdotL);
        diffuse += light->intensity * shadow * std::max(NdotL, 0.0f) * light_attenuation;
        specular += light->intensity * shadow * std::pow(std::max(normal.dot(half_v), 0.0f), p) * light_attenuation;
    }

    result_color = ka.cwiseProduct(ambient) + kd.cwiseProduct(diffuse) + ks.cwiseProduct(specular);
    return result_color * 255.f;
}