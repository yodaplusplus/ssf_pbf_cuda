#include "controls.hpp"
#include <glm/gtc/matrix_transform.hpp>

glm::mat4 controls::getViewMatrix(){
	return view_mat;
}
glm::mat4 controls::getProjectionMatrix(){
	return proj_mat;
}

controls::controls(GLFWwindow* arg_window) :
window(arg_window),
camera_pos(glm::vec3(2.0, 4.0, 10.0)),  // Initial position : on +Z
horizontal_angle(3.14f), // Initial horizontal angle : toward -Z
vertical_angle(0.f), // Initial vertical angle : none
initial_fov(45.0f),
speed(3.f), // 3 units / seconds
mouse_speed(0.0008f),
last_mouse_coord(0.f),
is_dragging(false),
near(0.1f),
far(50.f)
{
	int wind_width, wind_height;
	glfwGetWindowSize(window, &wind_width, &wind_height);
	// Ensure we can capture the escape key being pressed below
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glfwSetCursorPos(window, wind_width / 2, wind_height / 2);

	last_time = glfwGetTime();
}

void controls::computeMatricesFromInputs(){
	// Compute time difference between current and last frame
	double currentTime = glfwGetTime();
	float deltaTime = float(currentTime - last_time);

	// Get mouse position
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		if (is_dragging) {
			int wind_width, wind_height;
			glfwGetWindowSize(window, &wind_width, &wind_height);
			// Compute new orientation
			horizontal_angle -= mouse_speed * float(last_mouse_coord.x - xpos);
			vertical_angle -= mouse_speed * float(last_mouse_coord.y - ypos);
		}

		is_dragging = true;
	}
	else {
		is_dragging = false;
	}

	last_mouse_coord.x = xpos;
	last_mouse_coord.y = ypos;

	// Direction : Spherical coordinates to Cartesian coordinates conversion
	glm::vec3 direction(
		cos(vertical_angle) * sin(horizontal_angle), 
		sin(vertical_angle),
		cos(vertical_angle) * cos(horizontal_angle)
	);
	
	// Right vector
	glm::vec3 right = glm::vec3(
		sin(horizontal_angle - 3.14f/2.0f), 
		0,
		cos(horizontal_angle - 3.14f/2.0f)
	);
	
	// Up vector
	glm::vec3 up = glm::cross( right, direction );

	// Move forward
	if (glfwGetKey( window, GLFW_KEY_UP ) == GLFW_PRESS){
		camera_pos += direction * deltaTime * speed;
	}
	// Move backward
	if (glfwGetKey( window, GLFW_KEY_DOWN ) == GLFW_PRESS){
		camera_pos -= direction * deltaTime * speed;
	}
	// Strafe right
	if (glfwGetKey( window, GLFW_KEY_RIGHT ) == GLFW_PRESS){
		camera_pos += right * deltaTime * speed;
	}
	// Strafe left
	if (glfwGetKey( window, GLFW_KEY_LEFT ) == GLFW_PRESS){
		camera_pos -= right * deltaTime * speed;
	}

	float FoV = initial_fov;// - 5 * glfwGetMouseWheel(); // Now GLFW 3 requires setting up a callback for this. It's a bit too complicated for this beginner's tutorial, so it's disabled instead.

	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	proj_mat = glm::perspective(FoV, 4.0f / 3.0f, near, far);
	// Camera matrix
	view_mat = glm::lookAt(
		camera_pos,           // Camera is here
		camera_pos + direction, // and looks here : at the same camera_pos, plus "direction"
		up                  // Head is up (set to 0,-1,0 to look upside-down)
		);

	// For the next frame, the "last time" will be "now"
	last_time = currentTime;
}
