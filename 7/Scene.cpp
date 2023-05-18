//
// Created by Göksu Güvendiren on 2019-05-14.
//

#include "Scene.hpp"


void Scene::buildBVH() {
    printf(" - Generating BVH...\n\n");
    this->bvh = new BVHAccel(objects, 1, BVHAccel::SplitMethod::NAIVE);
}

Intersection Scene::intersect(const Ray &ray) const
{
    return this->bvh->Intersect(ray);
}

void Scene::sampleLight(Intersection &pos, float &pdf) const
{
    float emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
        }
    }
    float p = get_random_float() * emit_area_sum;
    emit_area_sum = 0;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        if (objects[k]->hasEmit()){
            emit_area_sum += objects[k]->getArea();
            if (p <= emit_area_sum){
                objects[k]->Sample(pos, pdf);
                break;
            }
        }
    }
}

bool Scene::trace(
        const Ray &ray,
        const std::vector<Object*> &objects,
        float &tNear, uint32_t &index, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearK = kInfinity;
        uint32_t indexK;
        Vector2f uvK;
        if (objects[k]->intersect(ray, tNearK, indexK) && tNearK < tNear) {
            *hitObject = objects[k];
            tNear = tNearK;
            index = indexK;
        }
    }


    return (*hitObject != nullptr);
}

Vector3f Scene::directLight(const Ray& ray, const Intersection& isect) const {
    Vector3f hitPoint = isect.coords;
    Vector3f hitNormal = normalize(isect.normal);

    float lightPdf;
    Intersection lightIsect;
    Scene::sampleLight(lightIsect, lightPdf);

    Vector3f lightDir = lightIsect.coords - hitPoint;
    float lightDist2 = dotProduct(lightDir, lightDir);
    lightDir = normalize(lightDir);

    Vector3f shadowPointOrigin = (dotProduct(ray.direction, hitNormal) < 0) ?
        hitPoint + hitNormal * EPSILON : hitPoint - hitNormal * EPSILON;

    Vector3f shadowDir = normalize(lightIsect.coords - shadowPointOrigin);
    Ray shadowRay = Ray(shadowPointOrigin, shadowDir);
    Intersection shadowIsect = Scene::intersect(shadowRay);

    bool shadow = true;
    if (shadowIsect.happened) {
        Vector3f diffVector = shadowIsect.coords - lightIsect.coords;
        float dist2 = dotProduct(diffVector, diffVector);
        if (dist2 < EPSILON) { shadow = false; }
    }
    if (shadow) { return 0.0; }

    Vector3f lightNormal = normalize(lightIsect.normal);
    float LdotN = std::max(0.f, dotProduct(lightDir, hitNormal));
    float LdotNN = std::max(0.f, dotProduct(-lightDir, lightNormal));
    Vector3f eval = isect.m->eval(lightDir, -ray.direction, hitNormal);

    return lightIsect.emit * eval * LdotN * LdotNN / (lightPdf * lightDist2);
}

Vector3f Scene::indirectLight(const Ray& ray, const Intersection& isect, int depth) const {
    if (get_random_float() > RussianRoulette) { return 0.0f; }

    Vector3f hitPoint = isect.coords;
    Vector3f hitNormal = normalize(isect.normal);

    Vector3f outDir = isect.m->sample(-ray.direction, hitNormal);
    Vector3f outOrigin = (dotProduct(ray.direction, hitNormal) < 0) ?
        hitPoint + hitNormal * EPSILON : hitPoint - hitNormal * EPSILON;
    Ray outRay(outOrigin, outDir);

    Vector3f eval = isect.m->eval(-ray.direction, outDir, hitNormal);
    float OdotN = std::max(0.f, dotProduct(outDir, hitNormal));
    float pdf = isect.m->pdf(-ray.direction, outDir, hitNormal);
    float scale = pdf < EPSILON ? 1 : OdotN / pdf;

    return castRay(outRay, depth + 1) * eval * scale / RussianRoulette;
}

// Implementation of Path Tracing
Vector3f Scene::castRay(const Ray& ray, int depth) const
{
    Intersection intersection = Scene::intersect(ray);
    if (!intersection.happened) { return Vector3f(0.0, 0.0, 0.0); }

    if (intersection.m->hasEmission() && depth == 0) {
        return intersection.m->getEmission();
    }
    Vector3f direct = directLight(ray, intersection);
    Vector3f indirect = indirectLight(ray, intersection, depth);
    return direct + indirect;
}