#pragma once
#include "../common/common_core.h"

#include <cstdint>
#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <memory>
#include <functional>

// used for uniform-grid neighbor search
struct hash_index_t {
	uint32_t* d_hash;
	uint32_t* d_index;
};

typedef float scalar_t;
typedef std::vector<glm::vec4> v4_vec;
typedef std::vector<glm::vec3> v3_vec;
typedef std::vector<glm::vec2> v2_vec;
typedef std::vector<scalar_t> v1_vec;

typedef glm::vec3 dom_dim;
typedef glm::uvec3 dom_udim;
typedef glm::ivec3 dom_idim;
typedef v3_vec dom_dim_vec;

struct pbf_phase
{
	dom_dim x;
	dom_dim v;
};

struct pbf_phase_array
{
	uint32_t max_num;
	uint32_t num;
	uint32_t current_step;
	dom_dim* x;
	dom_dim* v;
	void allocate(uint32_t num);	// must be called firstly, the number of elements, not bytes
	void free();
	void clear();
	void set(const pbf_phase_array& in);	// copy of in, deep copy
	pbf_phase_array& operator=(const pbf_phase_array& arg);
};

struct pbf_parameter
{
	scalar_t stable_density; // kg/m^3
	scalar_t stable_distance; // m
	scalar_t particle_mass; // kg 
	scalar_t smoothing_length; // m
	scalar_t time_step; // s
	scalar_t relaxation; // m^-2
	scalar_t xsph_parameter;
	scalar_t vc_parameter;
	pbf_parameter() :
		stable_density(-1.f), stable_distance(-1.f), particle_mass(-1.f), smoothing_length(-1.f), time_step(-1.f),
		relaxation(0.f), xsph_parameter(0.f), vc_parameter(0.f) {}
	void display();
};

namespace pbf {
	class neighbor_search;
}

struct pbf_external
{
	dom_dim body_force; // m/s^2
};

struct pbf_particle
{
	pbf_phase_array phase;
	std::shared_ptr<pbf::neighbor_search> ns;
	pbf_parameter parameter;
	pbf_external external;
	// utility method
	scalar_t current_time() const {
		return static_cast<scalar_t>(phase.current_step) * parameter.time_step;	// s
	}
	void allocate(uint32_t num, dom_dim domain_end);
	void free();
};

struct pbf_buffer
{
	pbf_phase_array interim;
	dom_dim* sorted_old_pos;
	dom_dim* sorted_predicted_pos;
	dom_dim* delta_position;
	scalar_t* scaling_factor;
	dom_dim* vorticity;
	scalar_t* kernels;
	dom_dim* grad_kernels;
	void allocate(uint32_t num, uint32_t max_pair_particle_num);	// must be called firstly, the number of elements, not bytes
	void free();
};
