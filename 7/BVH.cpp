#include <algorithm>
#include <cassert>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        node->area = objects[0]->getArea();
        return node;
    }
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
        return node;
    }
    else {
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                Union(centroidBounds, objects[i]->getBounds().Centroid());
        int dim = centroidBounds.maxExtent();
        switch (dim) {
        case 0:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().x <
                       f2->getBounds().Centroid().x;
            });
            break;
        case 1:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().y <
                       f2->getBounds().Centroid().y;
            });
            break;
        case 2:
            std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                return f1->getBounds().Centroid().z <
                       f2->getBounds().Centroid().z;
            });
            break;
        }

        auto beginning = objects.begin();
        auto middling = objects.begin() + (objects.size() / 2);
        auto ending = objects.end();

        auto leftshapes = std::vector<Object*>(beginning, middling);
        auto rightshapes = std::vector<Object*>(middling, ending);

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
        node->area = node->left->area + node->right->area;
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

std::pair<bool, Vector2f> inline boundIntersection(
    BVHBuildNode* node, const Ray& ray,
    const std::array<int, 3>& dirIsNeg, const Vector2f& range)
{
    Vector2f boundRange = range;
    node->bounds.IntersectP(ray, ray.direction_inv, dirIsNeg, boundRange);
    bool hit = boundRange.x <= boundRange.y + EPSILON;
    return std::make_pair(hit, boundRange);
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    Intersection isect;
    Vector2f range = { static_cast<float>(ray.t_min),
                       static_cast<float>(ray.t_max) };
    std::array<int, 3> dirIsNeg = {
        int(ray.direction.x < 0),
        int(ray.direction.y < 0),
        int(ray.direction.z < 0)
    };
    std::stack<BVHBuildNode*> nodeToVisit({ node });

    while (!nodeToVisit.empty()) {
        const auto cursor = nodeToVisit.top();
        nodeToVisit.pop();
        if (!boundIntersection(cursor, ray, dirIsNeg, range).first) { continue; }

        if (cursor->left != nullptr && cursor->right != nullptr) {
            auto leftRes = boundIntersection(cursor->left, ray, dirIsNeg, range);
            auto rightRes = boundIntersection(cursor->right, ray, dirIsNeg, range);

            if (!leftRes.first && rightRes.first) {
                nodeToVisit.push(cursor->right);
            }
            else if (leftRes.first && !rightRes.first) {
                nodeToVisit.push(cursor->left);
            }
            else if (leftRes.first && rightRes.first) {
                if (leftRes.second.x < rightRes.second.x) {
                    nodeToVisit.push(cursor->left);
                    nodeToVisit.push(cursor->right);
                }
                else {
                    nodeToVisit.push(cursor->right);
                    nodeToVisit.push(cursor->left);
                }
            }
        }
        else {
            auto objIsect = cursor->object[0].getIntersection(ray);
            if (!objIsect.happened) { continue; }
            if (objIsect.distance < range.y) {
                isect = objIsect;
                range.y = objIsect.distance;
            }
        }
    }
    return isect;
}


void BVHAccel::getSample(BVHBuildNode* node, float p, Intersection &pos, float &pdf){
    if(node->left == nullptr || node->right == nullptr){
        node->object->Sample(pos, pdf);
        pdf *= node->area;
        return;
    }
    if(p < node->left->area) getSample(node->left, p, pos, pdf);
    else getSample(node->right, p - node->left->area, pos, pdf);
}

void BVHAccel::Sample(Intersection &pos, float &pdf){
    float p = std::sqrt(get_random_float()) * root->area;
    getSample(root, p, pos, pdf);
    pdf /= root->area;
}