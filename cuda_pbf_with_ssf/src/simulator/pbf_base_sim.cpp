#include "pbf_base_sim.h"

namespace pbf {
;
using namespace std;

void pbf_base_sim::pbf_particle_alloc(uint32_t num_capacity, dom_dim domain_end)
{
	simulatee.allocate(num_capacity, domain_end);
	buffer.allocate(num_capacity, simulatee.ns->getMaxPairParticleNum());
}

std::pair<dom_dim, dom_dim> pbf_base_sim::getGlobalDomain() const
{
	return domain;
}

uint32_t pbf_base_sim::getElapsedStep() const
{
	auto current_step = simulatee.phase.current_step;
	return static_cast<uint32_t>(current_step);
}

scalar_t pbf_base_sim::getElapsedSeconds() const
{
	return simulatee.current_time();
}

scalar_t pbf_base_sim::getTimeStep() const
{
	return simulatee.parameter.time_step;
}

} // end of pbf ns
