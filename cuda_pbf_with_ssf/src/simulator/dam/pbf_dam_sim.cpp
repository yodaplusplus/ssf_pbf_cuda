#include "pbf_dam_sim.h"
#include "../../pbf/solver/pbf_solve.h"
#include "../../device_resource/swVBOs.h"
#include "../../pbf/util/pbf_cuda_util.h"

using namespace std;

namespace pbf {

pbf_dam_sim::pbf_dam_sim(scalar_t space)
{
	init_cond = make_shared<pbf_dam_init_cond>(space);
	init_cond->getParameter(simulatee.parameter);
	init_cond->getExternalForce(simulatee.external);
	auto domain = init_cond->getDomainRange();
	const auto num_capacity = 40000;
	simulatee.allocate(num_capacity, domain.second);
	buffer.allocate(num_capacity, simulatee.ns->getMaxPairParticleNum());
	vector<glm::vec3> x;
	vector<glm::vec3> v;
	init_cond->getDomainParticlePhaseHost(x, v);
	simulatee.phase.num = x.size();
	cout << "initial particle number: " << x.size() << endl;

#pragma region gl_interpo_init

	swVBOs::getInstance().enroll("pbf_particle");
	glBindBuffer(GL_ARRAY_BUFFER, swVBOs::getInstance().findVBO("pbf_particle"));
	glBufferData(GL_ARRAY_BUFFER, x.size() * sizeof(glm::vec3), x.data(), GL_STATIC_DRAW);
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
}

pbf_dam_sim::~pbf_dam_sim()
{
	simulatee.free();
	buffer.free();
}

void pbf_dam_sim::simulateOneStep()
{
	//simulatee.external.body_force

	cudaGraphicsMapResources(1, &cu_res);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	size_t pos_size;
	cudaGraphicsResourceGetMappedPointer((void**)&simulatee.phase.x, &pos_size, cu_res);
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
	one_step(simulatee, buffer, domain, 3);
	cudaGraphicsUnmapResources(1, &cu_res);
	simulatee.phase.x = NULL;
}

uint32_t pbf_dam_sim::getParticlesNum()
{
	return simulatee.phase.num;
}

void pbf_dam_sim::getParticlesPosition(std::vector<glm::vec3>& particle)
{
	const auto num = simulatee.phase.num;
	particle.resize(num);
	cudaMemcpy(particle.data(), simulatee.phase.x, num * sizeof(dom_dim), cudaMemcpyDeviceToHost);
}

GLuint pbf_dam_sim::getParticlesPositionVBO()
{
	return swVBOs::getInstance().findVBO("pbf_particle");
}

} // end of pbf ns
