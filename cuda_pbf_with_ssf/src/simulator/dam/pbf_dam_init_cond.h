#pragma once
#include "../../pbf/pbf_type.h"

namespace pbf {

class pbf_dam_init_cond
{
public:
	pbf_dam_init_cond(scalar_t space);
	~pbf_dam_init_cond() {};
	// space : desired particles space
	void getDomainParticlePhaseHost(dom_dim* pos, dom_dim* vel, uint32_t* particle_num) const;
	void getDomainParticlePhaseHost(std::vector<dom_dim>& pos, std::vector<dom_dim>& vel) const;
	// space : desired particles space
	void getDomainParticlePhaseDevice(dom_dim* pos, dom_dim* vel, uint32_t* particle_num) const;
	void getParameter(pbf_parameter& param);
	void getExternalForce(pbf_external& external);
	std::pair<dom_dim, dom_dim> getDomainRange() const;

private:
	pbf_parameter param;
	pbf_external external;
	dom_dim domain_origin;
	dom_dim domain_end;
	dom_dim fluid_origin;
	dom_dim fluid_end;
};

} // end of pbf ns
