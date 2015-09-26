#ifndef CONTROLS_HPP
#define CONTROLS_HPP

#include "common_core.h"

class controls
{
public:
	explicit controls(GLFWwindow* window);
	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();
	void computeMatricesFromInputs();
	const float near;
	const float far;

private:
	GLFWwindow* window;
	glm::mat4 view_mat;
	glm::mat4 proj_mat;
	glm::vec3 camera_pos;
	float horizontal_angle;
	float vertical_angle;
	float initial_fov;
	float speed;
	float mouse_speed;
	double last_time;
	glm::vec2 last_mouse_coord;
	bool is_dragging;
};

#endif