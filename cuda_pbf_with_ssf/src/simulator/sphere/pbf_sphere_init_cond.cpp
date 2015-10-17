#include "pbf_sphere_init_cond.h"
#include "../../pbf/util/pbf_arrangement.h"
#include "../../pbf/util/cuda/pbf_fill.h"

namespace pbf {
;
pbf_sphere_init_cond::pbf_sphere_init_cond(scalar_t space) :
domain_origin(dom_dim(0.1f)),
domain_end(dom_dim(7.f)),
fluid_origin(dom_dim(2.f)),
fluid_end(dom_dim(4.f))
{
	param.stable_density = 1.0e+3;
	param.stable_distance = space;
	const double particle_volume = pow(param.stable_distance, 3);
	param.particle_mass = particle_volume * param.stable_density;
	param.smoothing_length = 1.4f * param.stable_distance;
	param.time_step = 0.007f;
	param.relaxation = 2.f / (space * space); // notice that this unit is m^-2
	param.xsph_parameter = 0.1f;
	param.vc_parameter = 0.005f;

	external.body_force = dom_dim(0.f, -9.8f, 0.f);
	//external.body_force = dom_dim(0.f, 0.f, 0.f);

	std::cout << "relax: " << param.relaxation << std::endl;
}

namespace {
	void bodyH(dom_dim* pos, dom_dim* vel, uint32_t* num, scalar_t space, dom_dim body_origin, dom_dim body_end) {
		auto origin = body_origin;
		auto end = body_end;
		dom_dim_vec h_pos;
		cartesian_volume(h_pos, origin, end, space);
		memcpy(pos, h_pos.data(), h_pos.size() * sizeof(dom_dim));
		std::fill(vel, vel + h_pos.size(), dom_dim(0.f));
		*num = h_pos.size();
	}

	void bodyD(dom_dim* pos, dom_dim* vel, uint32_t* num, scalar_t space, dom_dim body_origin, dom_dim body_end) {
		auto origin = body_origin;
		auto end = body_end;
		dom_dim_vec h_pos;
		cartesian_volume(h_pos, origin, end, space);
		cudaMemcpy(pos, h_pos.data(), h_pos.size() * sizeof(dom_dim), cudaMemcpyHostToDevice);
		pbf::cuda::fill(vel, dom_dim(0.f), h_pos.size());
		*num = h_pos.size();
	}
}	// end of unnamed ns

void pbf_sphere_init_cond::getDomainParticlePhaseHost(dom_dim* pos, dom_dim* vel, uint32_t* particle_num) const
{
	bodyH(pos, vel, particle_num, param.stable_distance, fluid_origin, fluid_end);
}

void pbf_sphere_init_cond::getDomainParticlePhaseHost(std::vector<dom_dim>& pos, std::vector<dom_dim>& vel) const
{
	cartesian_volume(pos, fluid_origin, fluid_end, param.stable_distance);
	for (size_t i = 0; i < pos.size(); ++i) {
		vel.push_back(glm::vec3(0.f));
	}
}

void pbf_sphere_init_cond::getDomainParticlePhaseDevice(dom_dim* pos, dom_dim* vel, uint32_t* particle_num) const
{
	bodyD(pos, vel, particle_num, param.stable_distance, fluid_origin, fluid_end);
}

void pbf_sphere_init_cond::getParameter(pbf_parameter& arg_param)
{
	memcpy(&arg_param, &param, sizeof(pbf_parameter));
}

void pbf_sphere_init_cond::getExternalForce(pbf_external& arg_external)
{
	memcpy(&arg_external, &external, sizeof(pbf_external));
}

std::pair<dom_dim, dom_dim> pbf_sphere_init_cond::getDomainRange() const
{
	return std::make_pair(domain_origin, domain_end);
}

} // end of pbf ns