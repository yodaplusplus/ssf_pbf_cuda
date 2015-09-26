#include "pbf_type.h"
#include "util/pbf_cuda_util.h"
#include "interaction/pbf_neighbor_search.h"

void pbf_phase_array::allocate(uint32_t arg_num) {
	max_num = arg_num;
	num = 0;
	current_step = 0;
	cudaMalloc(&x, arg_num * sizeof(dom_dim));
	cudaMalloc(&v, arg_num * sizeof(dom_dim));
	cudaMemset(x, 0, arg_num * sizeof(dom_dim));
	cudaMemset(v, 0, arg_num * sizeof(dom_dim));
#ifndef NDEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void pbf_phase_array::free() {
	max_num = 0;
	num = 0;
	if (x != NULL) {
		cudaFree(x);
		x = NULL;
	}
	if (v != NULL) {
		cudaFree(v);
		v = NULL;
	}
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void pbf_phase_array::clear() {
	current_step = 0;
	num = 0;
}

void pbf_phase_array::set(const pbf_phase_array& in) {
	if (this->max_num < in.num) {
		exit(-1);
	}
	this->num = in.num;
	this->current_step = in.current_step;
	cudaMemcpy(this->x, in.x, in.num * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
	cudaMemcpy(this->v, in.v, in.num * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

pbf_phase_array& pbf_phase_array::operator=(const pbf_phase_array& arg) {
	this->num = arg.num;
	this->current_step = arg.num;
	cudaMemcpy(this->x, arg.x, arg.num * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
	cudaMemcpy(this->v, arg.v, arg.num * sizeof(dom_dim), cudaMemcpyDeviceToDevice);
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif

	return *this;
}

#define TO_STRING(var_name) #var_name

void pbf_parameter::display()
{
	using namespace std;
	cout << TO_STRING(stable_density) << ": " << stable_density << endl;
	cout << TO_STRING(stable_distance) << ": " << stable_distance << endl;
	cout << TO_STRING(smoothing_length) << ": " << smoothing_length << endl;
	cout << TO_STRING(particle_mass) << ": " << particle_mass << endl;
	cout << TO_STRING(time_step) << ": " << time_step << endl;
}

void pbf_buffer::allocate(uint32_t arg_num, uint32_t max_pair_particle_num)
{
	interim.allocate(arg_num);
	cudaMalloc(&sorted_old_pos, arg_num * sizeof(dom_dim));
	cudaMalloc(&sorted_predicted_pos, arg_num * sizeof(dom_dim));
	cudaMalloc(&delta_position, arg_num * sizeof(dom_dim));
	cudaMalloc(&scaling_factor, arg_num * sizeof(scalar_t));
	cudaMalloc(&vorticity, arg_num * sizeof(dom_dim));
	cudaMalloc(&kernels, arg_num * max_pair_particle_num * sizeof(scalar_t));
	cudaMalloc(&grad_kernels, arg_num * max_pair_particle_num * sizeof(dom_dim));
	cudaMemset(sorted_old_pos, 0, arg_num * sizeof(dom_dim));
	cudaMemset(sorted_predicted_pos, 0, arg_num * sizeof(dom_dim));
	cudaMemset(delta_position, 0, arg_num * sizeof(dom_dim));
	cudaMemset(scaling_factor, 0, arg_num*sizeof(scalar_t));
	cudaMemset(vorticity, 0, arg_num * sizeof(dom_dim));
	cudaMemset(kernels, 0, arg_num * max_pair_particle_num * sizeof(scalar_t));
	cudaMemset(grad_kernels, 0, arg_num * max_pair_particle_num * sizeof(dom_dim));
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

void pbf_buffer::free()
{
	interim.free();
	if (sorted_old_pos != NULL) {
		cudaFree(sorted_old_pos);
		sorted_old_pos = NULL;
	}
	if (sorted_predicted_pos != NULL) {
		cudaFree(sorted_predicted_pos);
		sorted_predicted_pos = NULL;
	}
	if (delta_position != NULL) {
		cudaFree(delta_position);
		delta_position = NULL;
	}
	if (scaling_factor != NULL) {
		cudaFree(scaling_factor);
		scaling_factor = NULL;
	}
	if(vorticity != NULL) {
		cudaFree(vorticity);
		vorticity = NULL;
	}
	if (kernels != nullptr) {
		cudaFree(kernels);
		kernels = nullptr;
	}
	if (grad_kernels != nullptr) {
		cudaFree(grad_kernels);
		grad_kernels = nullptr;
	}
#ifdef _DEBUG
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());
#endif
}

namespace {
dom_udim calcGridSize(double cell_size, dom_dim sim_end)
{
	dom_udim grid_size = dom_udim(1);
	while (true) {
		if (grid_size.x * cell_size > (sim_end.x) &&
			grid_size.y * cell_size > (sim_end.y) &&
			grid_size.z * cell_size > (sim_end.z)) {
			break;
		}
		auto x = grid_size.x * cell_size > (sim_end.x) ? grid_size.x : grid_size.x * 2;
		auto y = grid_size.y * cell_size > (sim_end.y) ? grid_size.y : grid_size.y * 2;
		auto z = grid_size.z * cell_size > (sim_end.z) ? grid_size.z : grid_size.z * 2;
		grid_size = dom_udim(x, y, z);

		if (grid_size.x > 1028 || grid_size.y > 1028 || grid_size.z > 1028) {
			std::cerr << "grid_size is too large" << std::endl;
			exit(-1);
		}
	}

	return grid_size;
}
}	// end of unnamed ns

void pbf_particle::allocate(uint32_t num, dom_dim domain_end)
{
	if (parameter.stable_density < 0.f || parameter.stable_distance < 0.f ||
		parameter.particle_mass < 0.f || parameter.smoothing_length < 0.f ||
		parameter.time_step < 0.f)
		exit(-1);
	phase.allocate(num);
	auto cell_width = parameter.smoothing_length;
	auto grid_size = calcGridSize(cell_width, domain_end);
	ns = std::make_shared<pbf::neighbor_search>(num, cell_width, grid_size);
}

void pbf_particle::free()
{
	phase.free();
}
