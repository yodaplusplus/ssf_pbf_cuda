#include "pbf_sphere_sim.h"
#include "../../pbf/solver/pbf_solve.h"
#include "../../device_resource/swVBOs.h"
#include "../../pbf/util/pbf_cuda_util.h"
#include "../../pbf/util/pbf_arrangement.h"
#include "../../pbf/util/cuda/pbf_add.h"

#include "glm\ext.hpp"

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

pbf_sphere_sim::pbf_sphere_sim(scalar_t space)
{
	init_cond = make_shared<pbf_sphere_init_cond>(space);
	init_cond->getParameter(simulatee.parameter);
	init_cond->getExternalForce(simulatee.external);
	auto domain = init_cond->getDomainRange();
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

	m_jet = make_shared<jet>();
	//m_jet->set(dom_dim(3.f, 3.f, 3.f), 0.8f, 0.2f, simulatee.parameter.stable_distance, dom_dim(1.f, 0.f, 0.f));
	m_jet->set(0.5f, glm::vec2(2.5f, 2.5f), glm::vec2(3.5f, 3.5f), simulatee.parameter.stable_distance, dom_dim(3.f, 0.f, 0.f));

	cudaGraphicsMapResources(1, &cu_res);
	size_t pos_size;
	cudaGraphicsResourceGetMappedPointer((void**)&simulatee.phase.x, &pos_size, cu_res);
	m_jet->add(simulatee);
	cudaGraphicsUnmapResources(1, &cu_res);
}

pbf_sphere_sim::~pbf_sphere_sim()
{
	simulatee.free();
	buffer.free();
}

void pbf_sphere_sim::simulateOneStep()
{
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

	cudaGraphicsMapResources(1, &cu_res);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	size_t pos_size;
	cudaGraphicsResourceGetMappedPointer((void**)&simulatee.phase.x, &pos_size, cu_res);

	if (cnt % 8 == 0)
		m_jet->add(simulatee);

#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	one_step(simulatee, buffer, 4);
	cudaGraphicsUnmapResources(1, &cu_res);
	simulatee.phase.x = NULL;
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

} // end of pbf ns
