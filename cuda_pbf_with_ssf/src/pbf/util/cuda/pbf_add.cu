#include "pbf_add.h"
#include <device_launch_parameters.h>


namespace {

}	// end of unnamed ns

namespace pbf {
namespace cuda {

void addParticle(
	pbf_particle& particle,
	const glm::vec3* adding_position,
	const glm::vec3* adding_velocity,
	uint32_t adding_num)
{
	auto x = particle.phase.x + particle.phase.num;
	auto v = particle.phase.v + particle.phase.num;
	cudaMemcpy(x, adding_position, adding_num * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
	cudaMemcpy(v, adding_velocity, adding_num * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
	particle.phase.num += adding_num;

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

}	// end of cuda ns
}	// end of pbf ns
