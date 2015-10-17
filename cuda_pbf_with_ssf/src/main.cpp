#include <iostream>
#include <memory>
#include <vector>
#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <glm\glm.hpp>

#include "common\shader.hpp"
#include "common\controls.hpp"
#include "common\renderer\ubo_arrangement.h"
#include "device_resource\swUBOs.h"
#include "device_resource\swVBOs.h"
#include "device_resource\swVAOs.h"
#include "device_resource\swShaders.h"
#include "device_resource\swFBOs.h"
#include "device_resource\swTextures.h"

#include "simulator\dam\pbf_dam_sim.h"
#include "simulator\sphere\pbf_sphere_sim.h"

using namespace std;

void error_callback(int error, const char* description)
{
	puts(description);
}

int main(void) {
	cout << "Hello, sir." << endl;

#pragma region window_init
	GLFWwindow* window;
	glm::ivec2 resolution_window(1280, 720);
#pragma region glfw_init
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_SAMPLES, 4);

	// Open a window and create its OpenGL context
	window = glfwCreateWindow(resolution_window.x, resolution_window.y, "Tutorial", NULL, NULL);
	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	glfwSetErrorCallback(error_callback);
#pragma endregion

#pragma region glew_init
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}
#pragma endregion

	printf("GL version: %s\n", glGetString(GL_VERSION));
	printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
#pragma endregion

#pragma region gl_state_init
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClearDepth(1.f);
	glEnable(GL_CULL_FACE);
	//glClearColor(1.f, 1.f, 1.f, 1.f);
	//glClearColor(0.f, 0.f, 0.f, 1.f);
	glClearColor(0.6f, 0.6f, 0.6f, 1.f);
#pragma endregion

	auto controller = std::make_shared<controls>(window);

	auto sim = std::make_shared<pbf::pbf_sphere_sim>(0.1f);
	//auto sim = std::make_shared<pbf::pbf_dam_sim>(0.16f);

#pragma region ubo_init
	swUBOs::getInstance().enroll("scene");
	const GLuint ubo_scene_binding = 5;
#pragma endregion

#pragma region ssf_object

#pragma region shader_init
	// Create and compile our GLSL program from the shaders
	const auto ssf_object_shader_id =
		LoadShaders("asset/shader/ssf_object.vert", "asset/shader/ssf_object.geom", "asset/shader/ssf_object.frag");
	const GLuint scene_block_index = glGetUniformBlockIndex(ssf_object_shader_id, "Scene");
	glUniformBlockBinding(ssf_object_shader_id, scene_block_index, ubo_scene_binding);
	swShaders::getInstance().enroll("ssf_object", ssf_object_shader_id);
#pragma endregion

#if 1
#pragma region vertices
	std::vector<glm::vec3> vertices;
	vertices.emplace_back(glm::vec3(0.f));
	vertices.emplace_back(glm::vec3(1.f));
	vertices.emplace_back(glm::vec3(1.f, 0.f, 0.f));
	vertices.emplace_back(glm::vec3(0.f, 1.f, 0.f));
	vertices.emplace_back(glm::vec3(0.f, 0.f, 1.f));
	vertices.emplace_back(glm::vec3(1.f, 1.f, 0.f));
	vertices.emplace_back(glm::vec3(0.f, 1.f, 1.f));
	vertices.emplace_back(glm::vec3(1.f, 0.f, 1.f));
#pragma endregion

#pragma region vbo_init
	swVBOs::getInstance().enroll("point_vertex");
	glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO("point_vertex"));
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
#pragma endregion
#endif

#pragma region vao_init
	swVAOs::getInstance().enroll("ssf_object");
	glBindVertexArray(swVAOs::getInstance().find("ssf_object"));
	glBindBuffer(GL_ARRAY_BUFFER, sim->getParticlesPositionVBO());
	glEnableVertexAttribArray(glGetAttribLocation(ssf_object_shader_id, "Position"));
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindVertexArray(0);
#pragma endregion

#pragma region tex_init
	swTextures::getInstance().enroll("tex_depth");
	glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_depth"));
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, resolution_window.x, resolution_window.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#pragma endregion

#pragma region fbo_init
	swFBOs::getInstance().enroll("ssf_object");
	glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_object"));
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, swTextures::getInstance().find("tex_depth"), 0);
	glDrawBuffer(GL_NONE);	// no color, only depth
#pragma endregion

#pragma endregion

#pragma region ssf_blur
	//const float blur_radius = 0.0000035f;
	const float blur_radius = 0.0000055f;

#pragma region shader_init
	const auto ssf_blur_shader_id = LoadShaders("asset/shader/ssf_blur.vert", "asset/shader/ssf_blur.frag");
	glUniformBlockBinding(ssf_blur_shader_id, glGetUniformBlockIndex(ssf_blur_shader_id, "Scene"), ubo_scene_binding);
	swShaders::getInstance().enroll("ssf_blur", ssf_blur_shader_id);
#pragma endregion

#pragma endregion

#pragma region ssf_blur_even

#pragma region tex_init
	swTextures::getInstance().enroll("tex_blur_even");
	glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_blur_even"));
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, resolution_window.x, resolution_window.y, 0, GL_R, GL_FLOAT, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, resolution_window.x, resolution_window.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#pragma endregion

#pragma region fbo_init
	swFBOs::getInstance().enroll("ssf_blur_even");
	glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_blur_even"));
	//glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, swTextures::getInstance().find("tex_blur"), 0);
	//glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, swTextures::getInstance().find("tex_blur_even"), 0);
	glDrawBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
#pragma endregion

#pragma endregion

#pragma region ssf_blur_odd

#pragma region tex_init
	swTextures::getInstance().enroll("tex_blur_odd");
	glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_blur_odd"));
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, resolution_window.x, resolution_window.y, 0, GL_R, GL_FLOAT, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, resolution_window.x, resolution_window.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#pragma endregion

#pragma region fbo_init
	swFBOs::getInstance().enroll("ssf_blur_odd");
	glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_blur_odd"));
	//glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, swTextures::getInstance().find("tex_blur"), 0);
	//glDrawBuffer(GL_COLOR_ATTACHMENT0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, swTextures::getInstance().find("tex_blur_odd"), 0);
	glDrawBuffer(GL_NONE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
#pragma endregion

#pragma endregion

#pragma region ssf_normal

#pragma region shader_init
	// Create and compile our GLSL program from the shaders
	const auto ssf_normal_shader_id = LoadShaders("asset/shader/ssf_normal.vert", "asset/shader/ssf_normal.frag");
	glUniformBlockBinding(ssf_normal_shader_id, glGetUniformBlockIndex(ssf_normal_shader_id, "Scene"), ubo_scene_binding);
	swShaders::getInstance().enroll("ssf_normal", ssf_normal_shader_id);
#pragma endregion

#pragma region tex_init
	swTextures::getInstance().enroll("tex_normal");
	glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_normal"));
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution_window.x, resolution_window.y, 0, GL_RGBA, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#pragma endregion

#pragma region fbo_init
	swFBOs::getInstance().enroll("ssf_normal");
	glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_normal"));
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, swTextures::getInstance().find("tex_normal"), 0);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
#pragma endregion

#pragma endregion

#pragma region ssf_shading

#pragma region shader_init
	// Create and compile our GLSL program from the shaders
	const auto ssf_shading_shader_id = LoadShaders("asset/shader/ssf_shading.vert", "asset/shader/ssf_shading.frag");
	glUniformBlockBinding(ssf_shading_shader_id, glGetUniformBlockIndex(ssf_shading_shader_id, "Scene"), ubo_scene_binding);
	swShaders::getInstance().enroll("ssf_shading", ssf_shading_shader_id);
#pragma endregion

#pragma region tex_init
	swTextures::getInstance().enroll("ssf_shading");
	glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("ssf_shading"));
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, resolution_window.x, resolution_window.y, 0, GL_RGBA, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#pragma endregion

#pragma region fbo_init
	swFBOs::getInstance().enroll("ssf_shading");
	glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_shading"));
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, swTextures::getInstance().find("ssf_shading"), 0);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);
#pragma endregion

#pragma endregion

#pragma region ssf_common

#pragma region fullscreen_quad
	static const GLfloat quad_vertex[] = {
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
	};
#pragma endregion

#pragma region vbo
	swVBOs::getInstance().enroll("fullscreen_vertex");
	glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO("fullscreen_vertex"));
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertex), quad_vertex, GL_STATIC_DRAW);
#pragma endregion

#pragma region vao
	swVAOs::getInstance().enroll("ssf_fullscreen");
	glBindVertexArray(swVAOs::getInstance().find("ssf_fullscreen"));
	glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO("fullscreen_vertex"));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindVertexArray(0);
#pragma endregion

#pragma endregion

	double lastTime = glfwGetTime();
	int nbFrames = 0;

	do {
#pragma region speed_measurement
		double currentTime = glfwGetTime();
		nbFrames++;
		if (currentTime - lastTime >= 1.0){ // If last prinf() was more than 1 sec ago
			// printf and reset timer
			printf("%f ms/frame\n", 1000.0 / double(nbFrames));
			printf("#particle: %d\n", sim->getParticlesNum());
			nbFrames = 0;
			lastTime += 1.0;
		}
#pragma endregion

#pragma region simulation
		{
			static bool is_stopped = false;
			if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
				is_stopped = is_stopped ? false : true;
			if (!is_stopped)
				sim->simulateOneStep();
		}
#pragma endregion

#pragma region vp_matrix
		controller->computeMatricesFromInputs();
		UBO::Scene ubo_scene;
		ubo_scene.proj = controller->getProjectionMatrix();
		ubo_scene.view = controller->getViewMatrix();
		ubo_scene.inv_proj = glm::inverse(ubo_scene.proj);
		const auto ubo_scene_id = swUBOs::getInstance().find("scene");
		glBindBuffer(GL_UNIFORM_BUFFER, ubo_scene_id);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UBO::Scene), &ubo_scene, GL_STATIC_DRAW);
#pragma endregion

#pragma region points_rendering
		{
			// state
			glDepthMask(GL_TRUE);
			// fbo
			glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_object"));
			// clear screen
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// shader
			const auto ssf_shader_id = swShaders::getInstance().find("ssf_object");
			glUseProgram(ssf_shader_id);
			// ubo
			glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
			// uniform per shader
			const auto world = glm::mat4(1.f);
			glUniformMatrix4fv(glGetUniformLocation(ssf_shader_id, "World"), 1, GL_FALSE, &world[0][0]);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "point_radius"), 0.18f);

			// draw
			glBindVertexArray(swVAOs::getInstance().find("ssf_object"));
			glDrawArrays(GL_POINTS, 0, sim->getParticlesNum());

			// restore state
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
#pragma endregion

#pragma region blur_map_even
		{
			// fbo
			glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_blur_even"));
			// clear screen
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// shader
			const auto ssf_shader_id = swShaders::getInstance().find("ssf_blur");
			glUseProgram(ssf_shader_id);
			// ubo
			glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
			// uniform per shader
			glUniform1f(glGetUniformLocation(ssf_shader_id, "proj_near"), controller->near);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "proj_far"), controller->far);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "blur_radius"), blur_radius);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "blur_scale"), resolution_window.y / tanf(45.f*0.5f*3.1415f / 180.f));
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_depth"));
			glUniform1i(glGetUniformLocation(ssf_shader_id, "tex_depth"), 0);
			// draw
			glBindVertexArray(swVAOs::getInstance().find("ssf_fullscreen"));
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// restore state
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
#pragma endregion

#pragma region blur_map_odd
		{
			// fbo
			glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_blur_odd"));
			// clear screen
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// shader
			const auto ssf_shader_id = swShaders::getInstance().find("ssf_blur");
			glUseProgram(ssf_shader_id);
			// ubo
			glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
			// uniform per shader
			glUniform1f(glGetUniformLocation(ssf_shader_id, "proj_near"), controller->near);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "proj_far"), controller->far);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "blur_radius"), blur_radius);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "blur_scale"), resolution_window.y / tanf(45.f*0.5f*3.1415f / 180.f));
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_blur_even"));
			glUniform1i(glGetUniformLocation(ssf_shader_id, "tex_depth"), 0);
			// draw
			glBindVertexArray(swVAOs::getInstance().find("ssf_fullscreen"));
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// restore state
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
#pragma endregion

#pragma region blur_map_even
		{
			// fbo
			glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_blur_even"));
			// clear screen
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// shader
			const auto ssf_shader_id = swShaders::getInstance().find("ssf_blur");
			glUseProgram(ssf_shader_id);
			// ubo
			glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
			// uniform per shader
			glUniform1f(glGetUniformLocation(ssf_shader_id, "proj_near"), controller->near);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "proj_far"), controller->far);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "blur_radius"), blur_radius);
			glUniform1f(glGetUniformLocation(ssf_shader_id, "blur_scale"), resolution_window.y / tanf(45.f*0.5f*3.1415f / 180.f));
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_blur_odd"));
			glUniform1i(glGetUniformLocation(ssf_shader_id, "tex_depth"), 0);
			// draw
			glBindVertexArray(swVAOs::getInstance().find("ssf_fullscreen"));
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// restore state
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
#pragma endregion

#pragma region normal_map
		{
			// fbo
			glBindFramebuffer(GL_FRAMEBUFFER, swFBOs::getInstance().find("ssf_normal"));
			// clear screen
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// shader
			const auto ssf_shader_id = swShaders::getInstance().find("ssf_normal");
			glUseProgram(ssf_shader_id);
			// ubo
			glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
			// uniform per shader
			glActiveTexture(GL_TEXTURE0);
			//glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_depth"));
			glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_blur_even"));
			glUniform1i(glGetUniformLocation(ssf_shader_id, "tex_depth"), 0);
			// draw
			glBindVertexArray(swVAOs::getInstance().find("ssf_fullscreen"));
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// restore state
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
		}
#pragma endregion

#pragma region shading
		{
			// clear screen
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// shader
			const auto ssf_shader_id = swShaders::getInstance().find("ssf_shading");
			glUseProgram(ssf_shader_id);
			// ubo
			glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
			// uniform per shader
			const auto world = glm::mat4(1.f);
			glUniformMatrix4fv(glGetUniformLocation(ssf_shader_id, "World"), 1, GL_FALSE, &world[0][0]);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_depth"));
			glUniform1i(glGetUniformLocation(ssf_shader_id, "tex_depth"), 0);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, swTextures::getInstance().find("tex_normal"));
			glUniform1i(glGetUniformLocation(ssf_shader_id, "tex_normal"), 1);
			// draw
			glBindVertexArray(swVAOs::getInstance().find("ssf_fullscreen"));
			glDrawArrays(GL_TRIANGLES, 0, 6);
			// clear
			//glBindVertexArray(0);
		}
#pragma endregion

		sim->drawNeighborSearchArea(ubo_scene_id);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}// Check if the ESC key was pressed or the window was closed
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

	swUBOs::getInstance().dismiss();
	swVAOs::getInstance().dismiss();
	swVBOs::getInstance().dismiss();
	swShaders::getInstance().dismiss();
	swFBOs::getInstance().dismiss();
	swTextures::getInstance().dismiss();

	glfwTerminate();

	return 0;
}
