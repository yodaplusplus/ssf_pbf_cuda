#include "pbf_update.h"
#include "../../interaction/cuda/pbf_contribution.h"
#include "../../kernel/cuda/pbf_kernel.h"
#include "../../util/pbf_cuda_util.h"
#include "../../interaction/cuda/pbf_grid.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>

using namespace std;

namespace pbf {
namespace cuda {
;
void updatePosition(
	dom_dim* new_position,
	const dom_dim* result_position,
	int num_particle
	)
{
	cudaMemcpy(new_position, result_position, num_particle * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
}

__global__ void updateVelocityCUDA(
	dom_dim* velocity,
	const dom_dim* new_position,
	const dom_dim* old_position,
	scalar_t inv_time_step,
	int num_particle)
{
	uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= num_particle) return;

	auto new_pos = new_position[gid];
	auto old_pos = old_position[gid];

	velocity[gid] = inv_time_step * (new_pos - old_pos);
}

void updateVelocity(
	dom_dim* velocity,
	const dom_dim* new_position,
	const dom_dim* old_position,
	scalar_t inv_time_step,
	int num_particle
	)
{
	if (num_particle > 0) {
		uint32_t num_thread, num_block;
		computeGridSize(num_particle, 128, num_block, num_thread);
		updateVelocityCUDA<< < num_block, num_thread >> >(velocity, new_position, old_position, inv_time_step, num_particle);
	}

#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void updateVelocity(
	dom_dim* new_vel,
	const dom_dim* result_vel,
	int num_particle
	)
{
	cudaMemcpy(new_vel, result_vel, num_particle * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
}

}	// end of cuda ns
}	// end of pbf ns