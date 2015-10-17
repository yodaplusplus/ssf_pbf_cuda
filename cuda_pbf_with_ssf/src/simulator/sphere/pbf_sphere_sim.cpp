#include "pbf_sphere_sim.h"
#include "../../pbf/solver/pbf_solve.h"
#include "../../device_resource/swVBOs.h"
#include "../../pbf/util/pbf_cuda_util.h"
#include "../../pbf/util/pbf_arrangement.h"
#include "../../pbf/util/cuda/pbf_add.h"

#include "glm\ext.hpp"

#include "../../common/shader.hpp"
#include "../../device_resource/swShaders.h"
#include "../../device_resource/swVBOs.h"
#include "../../device_resource/swIBOs.h"
#include "../../device_resource/swVAOs.h"

using namespace std;

namespace pbf {

class pbf_sphere_sim::jet {
public:
	jet() : d_pos(nullptr), d_vel(nullptr) {}
	~jet() {
		cudaFree(d_pos);
		cudaFree(d_vel);
	}
	// cylinder
	void set(dom_dim center, scalar_t radius, scalar_t height, scalar_t space, dom_dim init_vel) {
		if (d_pos != nullptr) cudaFree(d_pos);
		if (d_vel != nullptr) cudaFree(d_vel);

		h_pos.clear();
		cartesian_cylinder(h_pos, std::string("x"), center.x, glm::vec2(center.y, center.z), radius, height, space);
		h_vel.clear();
		h_vel.resize(h_pos.size());
		fill(h_vel.begin(), h_vel.end(), init_vel);

		cudaMalloc(&d_pos, h_pos.size() * sizeof(dom_dim));
		cudaMalloc(&d_vel, h_pos.size() * sizeof(dom_dim));

		cudaMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(dom_dim), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vel, h_vel.data(), h_pos.size() * sizeof(dom_dim), cudaMemcpyHostToDevice);

#ifdef _DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	// rectangle
	void set(scalar_t axis_value, glm::vec2 origin, glm::vec2 end, scalar_t space, dom_dim init_vel) {
		if (d_pos != nullptr) cudaFree(d_pos);
		if (d_vel != nullptr) cudaFree(d_vel);

		h_pos.clear();
		cartesian_rectangle(h_pos, std::string("x"), axis_value, origin, end, space);
		h_vel.clear();
		h_vel.resize(h_pos.size());
		fill(h_vel.begin(), h_vel.end(), init_vel);

		cudaMalloc(&d_pos, h_pos.size() * sizeof(dom_dim));
		cudaMalloc(&d_vel, h_pos.size() * sizeof(dom_dim));

		cudaMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(dom_dim), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vel, h_vel.data(), h_pos.size() * sizeof(dom_dim), cudaMemcpyHostToDevice);

#ifdef _DEBUG
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());
#endif
	}

	void add(pbf_particle& particle) {
		cuda::addParticle(particle, d_pos, d_vel, h_pos.size());
	}

private:
	dom_dim_vec h_pos;
	dom_dim_vec h_vel;
	dom_dim* d_pos;
	dom_dim* d_vel;
};

class pbf_sphere_sim::neighbor_search_area {
public:
	neighbor_search_area(std::pair<dom_dim, dom_dim>& area) : ubo_scene_binding(5) {

#pragma region shader
		const auto constant_model_shader_id =
			LoadShaders("asset/shader/constant_model.vert", "asset/shader/constant_model.frag");
		const GLuint scene_block_index = glGetUniformBlockIndex(constant_model_shader_id, "Scene");
		glUniformBlockBinding(constant_model_shader_id, scene_block_index, ubo_scene_binding);
		swShaders::getInstance().enroll("neighbor_search_area", constant_model_shader_id);
#pragma endregion

#pragma region vbo
		const GLfloat cube_vertices[] = {
			// front
			area.first.x, area.first.y, area.second.z,
			area.second.x, area.first.y, area.second.z,
			area.second.x, area.second.y, area.second.z,
			area.first.x, area.second.y, area.second.z,
			// back
			area.first.x, area.first.y, area.first.z,
			area.second.x, area.first.y, area.first.z,
			area.second.x, area.second.y, area.first.z,
			area.first.x, area.second.y, area.first.z,
		};
		swVBOs::getInstance().enroll("neighbor_search_area");
		glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO("neighbor_search_area"));
		glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
#pragma endregion

#pragma region ibo
		const GLushort cube_indices[] = {
			// front
			0, 1, 2,
			2, 3, 0,
			// top
			3, 2, 6,
			6, 7, 3,
			// back
			7, 6, 5,
			5, 4, 7,
			// bottom
			4, 5, 1,
			1, 0, 4,
			// left
			4, 0, 3,
			3, 7, 4,
			// right
			1, 5, 6,
			6, 2, 1,
		};
		swIBOs::getInstance().enroll("neighbor_search_area");
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, swIBOs::getInstance().findIBO("neighbor_search_area"));
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
#pragma endregion

#pragma region vao
		swVAOs::getInstance().enroll("neighbor_search_area");
		glBindVertexArray(swVAOs::getInstance().find("neighbor_search_area"));
		glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO("neighbor_search_area"));
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glBindVertexArray(0);
#pragma endregion

	}
	~neighbor_search_area() {}
	void draw(GLuint ubo_scene_id) {
		//const std::pair<dom_dim, dom_dim>& area
		//std::vector<dom_dim> vertices;
		//vertices.emplace_back(area.first);

		// clear screen
		//glClear(GL_DEPTH_BUFFER_BIT);
		// state
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDisable(GL_CULL_FACE);
		// shader
		const auto shader_id = swShaders::getInstance().find("neighbor_search_area");
		glUseProgram(shader_id);
		// ubo
		glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
		// uniform per shader
		const auto world = glm::mat4(1.f);
		glUniformMatrix4fv(glGetUniformLocation(shader_id, "World"), 1, GL_FALSE, &world[0][0]);
		// draw
		glBindVertexArray(swVAOs::getInstance().find("neighbor_search_area"));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, swIBOs::getInstance().findIBO("neighbor_search_area"));
		glDrawElements(GL_TRIANGLES, 3 * 2 * 6, GL_UNSIGNED_SHORT, nullptr);
		// restore state
		glUseProgram(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_CULL_FACE);
	}
private:
	const GLuint ubo_scene_binding;
};

class pbf_sphere_sim::area_drawing {
public:
	area_drawing(const string& name, std::pair<dom_dim, dom_dim>& area) : obj_name(name), ubo_scene_binding(5) {

#pragma region shader
		const auto constant_model_shader_id =
			LoadShaders("asset/shader/constant_model.vert", "asset/shader/constant_model.frag");
		const GLuint scene_block_index = glGetUniformBlockIndex(constant_model_shader_id, "Scene");
		glUniformBlockBinding(constant_model_shader_id, scene_block_index, ubo_scene_binding);
		swShaders::getInstance().enroll(std::string("area_drawing") + obj_name, constant_model_shader_id);
#pragma endregion

#pragma region vbo
		const GLfloat cube_vertices[] = {
			// front
			area.first.x, area.first.y, area.second.z,
			area.second.x, area.first.y, area.second.z,
			area.second.x, area.second.y, area.second.z,
			area.first.x, area.second.y, area.second.z,
			// back
			area.first.x, area.first.y, area.first.z,
			area.second.x, area.first.y, area.first.z,
			area.second.x, area.second.y, area.first.z,
			area.first.x, area.second.y, area.first.z,
		};
		swVBOs::getInstance().enroll(std::string("area_drawing") + obj_name);
		glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO(std::string("area_drawing") + obj_name));
		glBufferData(GL_ARRAY_BUFFER, sizeof(cube_vertices), cube_vertices, GL_STATIC_DRAW);
#pragma endregion

#pragma region ibo
		const GLushort cube_indices[] = {
			// front
			0, 1, 2,
			2, 3, 0,
			// top
			3, 2, 6,
			6, 7, 3,
			// back
			7, 6, 5,
			5, 4, 7,
			// bottom
			4, 5, 1,
			1, 0, 4,
			// left
			4, 0, 3,
			3, 7, 4,
			// right
			1, 5, 6,
			6, 2, 1,
		};
		swIBOs::getInstance().enroll(std::string("area_drawing") + obj_name);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, swIBOs::getInstance().findIBO(std::string("area_drawing") + obj_name));
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_indices), cube_indices, GL_STATIC_DRAW);
#pragma endregion

#pragma region vao
		swVAOs::getInstance().enroll(std::string("area_drawing") + obj_name);
		glBindVertexArray(swVAOs::getInstance().find(std::string("area_drawing") + obj_name));
		glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO(std::string("area_drawing") + obj_name));
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glBindVertexArray(0);
#pragma endregion

	}
	~area_drawing() {}
	void draw(GLuint ubo_scene_id) {
		//const std::pair<dom_dim, dom_dim>& area
		//std::vector<dom_dim> vertices;
		//vertices.emplace_back(area.first);

		// clear screen
		//glClear(GL_DEPTH_BUFFER_BIT);
		// state
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glDisable(GL_CULL_FACE);
		// shader
		const auto shader_id = swShaders::getInstance().find(std::string("area_drawing") + obj_name);
		glUseProgram(shader_id);
		// ubo
		glBindBufferBase(GL_UNIFORM_BUFFER, ubo_scene_binding, ubo_scene_id);
		// uniform per shader
		const auto world = glm::mat4(1.f);
		glUniformMatrix4fv(glGetUniformLocation(shader_id, "World"), 1, GL_FALSE, &world[0][0]);
		// draw
		glBindVertexArray(swVAOs::getInstance().find(std::string("area_drawing") + obj_name));
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, swIBOs::getInstance().findIBO(std::string("area_drawing") + obj_name));
		glDrawElements(GL_TRIANGLES, 3 * 2 * 6, GL_UNSIGNED_SHORT, nullptr);
		// restore state
		glUseProgram(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_CULL_FACE);
	}
private:
	const std::string obj_name;
	const GLuint ubo_scene_binding;
};

pbf_sphere_sim::pbf_sphere_sim(scalar_t space)
{
	init_cond = make_shared<pbf_sphere_init_cond>(space);
	init_cond->getParameter(simulatee.parameter);
	init_cond->getExternalForce(simulatee.external);
	domain = init_cond->getDomainRange();
	const auto num_capacity = 60000;
	simulatee.allocate(num_capacity, domain.second);
	buffer.allocate(num_capacity, simulatee.ns->getMaxPairParticleNum());
	vector<glm::vec3> x(0);
	vector<glm::vec3> v(0);
	//init_cond->getDomainParticlePhaseHost(x, v);
	simulatee.phase.num = x.size();
	cout << "initial particle number: " << x.size() << endl;
	
#pragma region gl_interpo_init

	swVBOs::getInstance().enroll("pbf_particle");
	glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO("pbf_particle"));
	//glBufferData(GL_ARRAY_BUFFER, x.size() * sizeof(glm::vec3), x.data(), GL_STATIC_DRAW);
	glBufferData(GL_ARRAY_BUFFER, num_capacity * sizeof(glm::vec3), nullptr, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, x.size() * sizeof(glm::vec3), x.data());
	gpuErrchk(cudaDeviceSynchronize());
	cudaGraphicsGLRegisterBuffer(&cu_res, swVBOs::getInstance().findVBO("pbf_particle"), cudaGraphicsMapFlagsNone);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	// modify particles data
	cudaFree(simulatee.phase.x);
	cudaMemcpy(simulatee.phase.v, v.data(), v.size() * sizeof(dom_dim), cudaMemcpyHostToDevice);
#pragma endregion

#pragma region water_jet_init
	m_jet = make_shared<jet>();
	//m_jet->set(dom_dim(3.f, 3.f, 3.f), 0.8f, 0.2f, simulatee.parameter.stable_distance, dom_dim(1.f, 0.f, 0.f));
	m_jet->set(1.25f, glm::vec2(3.5f, 2.5f), glm::vec2(4.5f, 3.5f), simulatee.parameter.stable_distance, dom_dim(2.8f, 0.f, 0.f));

	cudaGraphicsMapResources(1, &cu_res);
	size_t pos_size;
	cudaGraphicsResourceGetMappedPointer((void**)&simulatee.phase.x, &pos_size, cu_res);
	m_jet->add(simulatee);
	cudaGraphicsUnmapResources(1, &cu_res);
#pragma endregion

#pragma region area_drawing
	// draw neighbor search area for debug
	auto grid_size = simulatee.ns->getGridSize();
	auto cell_width = simulatee.ns->getCellWidth();
	auto area = make_pair(dom_dim(0.f, 0.f, 0.f), dom_dim(grid_size.x, grid_size.y, grid_size.z) * cell_width * 2.f);
	m_nsa = make_shared<area_drawing>("neighbor_search", area);
	// draw simulation area for debug
	m_sim_area = make_shared<area_drawing>("simulation", domain);
#pragma endregion

}

pbf_sphere_sim::~pbf_sphere_sim()
{
	simulatee.free();
	buffer.free();
}

void pbf_sphere_sim::simulateOneStep()
{
#pragma region gravity_control
	static glm::vec3 g(0.f, -9.8f, 0.f);
	static float rad = 0.f;
	static float delta = 0.015f;
	static uint32_t cnt = 0;
	//simulatee.external.body_force = glm::rotateZ(g, rad);
	rad += delta;
	if (rad > glm::pi<float>() * 2.f) {
		rad = 0.f;
		//delta = delta * 1.1f;
	}
	if (cnt > 100) {
		//simulatee.external.body_force = glm::rotateZ(g, rad);
	}
	else {
		simulatee.external.body_force = g;
		rad = 0.f;
	}
	++cnt;
#pragma endregion

	cudaGraphicsMapResources(1, &cu_res);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	size_t pos_size;
	cudaGraphicsResourceGetMappedPointer((void**)&simulatee.phase.x, &pos_size, cu_res);

	if (cnt % 8 == 0 && simulatee.phase.num < simulatee.phase.max_num * 0.8)
		m_jet->add(simulatee);

	one_step(simulatee, buffer, domain, 3);
	cudaGraphicsUnmapResources(1, &cu_res);
	simulatee.phase.x = NULL;

#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

uint32_t pbf_sphere_sim::getParticlesNum()
{
	return simulatee.phase.num;
}

void pbf_sphere_sim::getParticlesPosition(std::vector<glm::vec3>& particle)
{
	const auto num = simulatee.phase.num;
	particle.resize(num);
	cudaMemcpy(particle.data(), simulatee.phase.x, num * sizeof(dom_dim), cudaMemcpyDeviceToHost);
}

GLuint pbf_sphere_sim::getParticlesPositionVBO()
{
	return swVBOs::getInstance().findVBO("pbf_particle");
}

void pbf_sphere_sim::drawNeighborSearchArea(GLuint ubo_scene_id)
{
	m_nsa->draw(ubo_scene_id);
	m_sim_area->draw(ubo_scene_id);
}

} // end of pbf ns
