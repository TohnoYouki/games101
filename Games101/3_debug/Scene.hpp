#pragma once

#include <mutex>
#include "global.hpp"
#include <Eigen/Eigen>
#include "Shader.hpp"
#include "Triangle.hpp"
#include "Texture.hpp"

std::vector<Triangle> load_mesh(std::string path);
std::vector<Triangle> plane(float width);

class Transform
{
public:
	Eigen::Matrix4f matrix;

	Transform(float scale = 0.0f, Eigen::Vector3f position = Eigen::Vector3f(), 
			  Eigen::Vector3f angle = Eigen::Vector3f());

	Eigen::Matrix4f get_model_matrix() { return matrix; };

	inline Eigen::Vector3f position() { return matrix.block<3, 1>(0, 3); }

	void translate(Eigen::Vector3f position);

	void rotate(Eigen::Vector3f angle);
};

class MeshObject
{
public:
	Transform transform;
	std::vector<Triangle>* mesh;
	std::function<Eigen::Vector3f(vertex_shader_payload)> vertex_shader;
	std::function<Eigen::Vector3f(fragment_shader_payload)> fragment_shader;
};

class Camera
{
public:
	Transform transform;
	float eye_fov;
	float aspect_ratio;
	float zNear, zFar;

	Eigen::Matrix4f get_view_matrix() {
		return transform.get_model_matrix().inverse();
	}
	Eigen::Matrix4f get_projection_matrix();

	inline const Eigen::Vector3f& position() {
		return transform.position();
	}
};

class PointLight
{
public:
	
	Camera camera;
	Eigen::Vector3f intensity;

	int width, height;
	std::vector<float> shadowmaps[6];
	Eigen::Matrix4f shadow_view_rotations[6];
	float lightsize = 10.0;

	PointLight(Camera camera, Eigen::Vector3f intensity, int width, int height);

	inline const Eigen::Vector3f& position() {
		return camera.transform.position();
	}

	Eigen::Matrix4f shadow_view_matrix(int id);
	float sample_shadowmap(Eigen::Vector3f vector, float NdotL);
	float pcss(Eigen::Vector3f vector, float NdotL);
};

class Scene
{
public:
	std::mutex mutex;
	int object_index = 0;
	int camera_index = 0;

	Camera camera;
	std::vector<Texture> textures;
	std::vector<std::vector<Triangle>> meshes;
	std::vector<MeshObject> objects;
	std::vector<PointLight> lights;
	std::vector<Camera*> pcameras;
};