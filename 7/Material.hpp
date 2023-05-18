//
// Created by LEI XU on 5/16/19.
//

#ifndef RAYTRACING_MATERIAL_H
#define RAYTRACING_MATERIAL_H

#include "Vector.hpp"

enum MaterialType { DIFFUSE, MICROFACET };

class Material{
private:

    // Compute reflection direction
    Vector3f reflect(const Vector3f &I, const Vector3f &N) const
    {
        return I - 2 * dotProduct(I, N) * N;
    }

    // Compute refraction direction using Snell's law
    //
    // We need to handle with care the two possible situations:
    //
    //    - When the ray is inside the object
    //
    //    - When the ray is outside.
    //
    // If the ray is outside, you need to make cosi positive cosi = -N.I
    //
    // If the ray is inside, you need to invert the refractive indices and negate the normal N
    Vector3f refract(const Vector3f &I, const Vector3f &N, const float &ior) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        Vector3f n = N;
        if (cosi < 0) { cosi = -cosi; } else { std::swap(etai, etat); n= -N; }
        float eta = etai / etat;
        float k = 1 - eta * eta * (1 - cosi * cosi);
        return k < 0 ? 0 : eta * I + (eta * cosi - sqrtf(k)) * n;
    }

    // Compute Fresnel equation
    //
    // \param I is the incident view direction
    //
    // \param N is the normal at the intersection point
    //
    // \param ior is the material refractive index
    //
    // \param[out] kr is the amount of light reflected
    void fresnel(const Vector3f &I, const Vector3f &N, const float &ior, float &kr) const
    {
        float cosi = clamp(-1, 1, dotProduct(I, N));
        float etai = 1, etat = ior;
        if (cosi > 0) {  std::swap(etai, etat); }
        // Compute sini using Snell's law
        float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
        // Total internal reflection
        if (sint >= 1) {
            kr = 1;
        }
        else {
            float cost = sqrtf(std::max(0.f, 1 - sint * sint));
            cosi = fabsf(cosi);
            float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
            float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
            kr = (Rs * Rs + Rp * Rp) / 2;
        }
        // As a consequence of the conservation of energy, transmittance is given by:
        // kt = 1 - kr;
    }

    Vector3f toWorld(const Vector3f &a, const Vector3f &N){
        Vector3f B, C;
        if (std::fabs(N.x) > std::fabs(N.y)){
            float invLen = 1.0f / std::sqrt(N.x * N.x + N.z * N.z);
            C = Vector3f(N.z * invLen, 0.0f, -N.x *invLen);
        }
        else {
            float invLen = 1.0f / std::sqrt(N.y * N.y + N.z * N.z);
            C = Vector3f(0.0f, N.z * invLen, -N.y *invLen);
        }
        B = crossProduct(C, N);
        return a.x * B + a.y * C + a.z * N;
    }

public:
    MaterialType m_type;
    //Vector3f m_color;
    Vector3f m_emission;
    float ior;
    float alpha;
    Vector3f Kd, Ks;
    float specularExponent;
    //Texture tex;

    inline Material(MaterialType t=DIFFUSE, Vector3f e=Vector3f(0,0,0));
    inline MaterialType getType();
    //inline Vector3f getColor();
    inline Vector3f getColorAt(double u, double v);
    inline Vector3f getEmission();
    inline bool hasEmission();
    inline float microfacetDistribution(float costheta);
    inline float microfacetLambda(float costheta);
    inline float microfacetShadowMask(float costhetai, float costhetao);

    // sample a ray by Material properties
    inline Vector3f sample(const Vector3f &wi, const Vector3f &N);
    // given a ray, calculate the PdF of this ray
    inline float pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N);
    // given a ray, calculate the contribution of this ray
    inline Vector3f eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N);

};

Material::Material(MaterialType t, Vector3f e){
    m_type = t;
    //m_color = c;
    m_emission = e;
}

MaterialType Material::getType(){return m_type;}
///Vector3f Material::getColor(){return m_color;}
Vector3f Material::getEmission() {return m_emission;}
bool Material::hasEmission() {
    if (m_emission.norm() > EPSILON) return true;
    else return false;
}

float Material::microfacetDistribution(float costheta)
{
    double costheta2 = std::pow(costheta, 2);
    double invalpha2 = std::pow(1.0 / alpha, 2);
    double value = (costheta2 + (1 - costheta2) * invalpha2);
    return static_cast <float>(1.0 * invalpha2 / (M_PI * value * value));
}

float Material::microfacetLambda(float costheta) {
    double costheta2 = costheta * costheta;
    double sintheta2 = 1 - costheta2;
    double tantheta2 = sintheta2 / costheta2;
    return (-1 + std::sqrt(1. + tantheta2 * alpha * alpha)) / 2;
}

float Material::microfacetShadowMask(float costhetai, float costhetao)
{
    return 1.0 / (1 + microfacetLambda(costhetai) + microfacetLambda(costhetao));
}


Vector3f Material::getColorAt(double u, double v) {
    return Vector3f();
}


Vector3f Material::sample(const Vector3f &wi, const Vector3f &N){
    switch(m_type){
        case DIFFUSE:
        {
            // uniform sample on the hemisphere
            float x_1 = get_random_float(), x_2 = get_random_float();

            float z = std::sqrt((1.0f - x_1));
            float phi = phi = 2 * M_PI * x_2;
            float r = std::sqrt(x_1);
            /*
            float z = std::fabs(1.0f - 2.0f * x_1);
            float r = std::sqrt(1.0f - z * z), phi = 2 * M_PI * x_2;*/
            Vector3f localRay(r*std::cos(phi), r*std::sin(phi), z);

            return toWorld(localRay, N);
            
            break;
        }
        case MICROFACET:
        {
            float x_1 = get_random_float(), x_2 = get_random_float();
            double alpha2 = alpha * alpha;
            double costheta2 = x_1 / (alpha2 - (alpha2 - 1) * x_1);
            float z = std::sqrt(costheta2);
            float r = std::sqrt(1 - costheta2);
            float phi = phi = 2 * M_PI * x_2;
            Vector3f h(r * std::cos(phi), r * std::sin(phi), z);
            h = toWorld(h, N);
            return reflect(wi, h);

            break;
        }
    }
}

float Material::pdf(const Vector3f &wi, const Vector3f &wo, const Vector3f &N){
    switch(m_type){
        case DIFFUSE:
        {
            // uniform sample probability 1 / (2 * PI)
            if (dotProduct(wo, N) > 0.0f)
                //return 0.5f / M_PI;
                return dotProduct(wo, N) / M_PI;
            else
                return 0.0f;
            break;
        }
        case MICROFACET:
        {
            Vector3f h = normalize(wo + wi);
            float costheta = dotProduct(h, N);
            return microfacetDistribution(costheta) * costheta / (4 * dotProduct(wi, h));
            break;
        }
    }
}

Vector3f Material::eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N){
    switch(m_type){
        case DIFFUSE:
        {
            // calculate the contribution of diffuse   model
            float cosalpha = dotProduct(N, wo);
            if (cosalpha > 0.0f) {
                Vector3f diffuse = Kd / M_PI;
                return diffuse;
            }
            else
                return Vector3f(0.0f);
            break;
        }
        case MICROFACET:
        {
            float cosThetai = std::max(0.0f, dotProduct(wi, N));
            float cosThetao = std::max(0.0f, dotProduct(wo, N));
            Vector3f h = (wi + wo) / 2;
            if (cosThetai <= 0.0f || cosThetao <= 0.0f) { return Vector3f(0.0f); }
            if (h.x == 0 && h.y == 0 && h.z == 0) { return Vector3f(0.0f); }
            h = normalize(h);

            float f;
            fresnel(wi, h, ior, f);
            float d = microfacetDistribution(dotProduct(h, N));
            float g = microfacetShadowMask(cosThetai, cosThetao);
            return Kd * f * d * g / (4 * cosThetai * cosThetao);
            
        }
    }
}

#endif //RAYTRACING_MATERIAL_H
