#pragma once
#include "../pbf/pbf.h"

namespace pbf {
;
class pbf_base_sim
{
public:
	pbf_base_sim() {}
	virtual ~pbf_base_sim() {}
	// <(xmin, ymin, zmin), (xmax, ymax, zmax)>
	std::pair<dom_dim, dom_dim> getGlobalDomain() const;
	uint32_t getElapsedStep() const;
	scalar_t getElapsedSeconds() const;
	scalar_t getTimeStep() const;

protected:
	// domain information
	std::pair<dom_dim, dom_dim> domain; // <(xmin, ymin, zmin), (xmax, ymax, zmax)>
	// particles data
	pbf_particle simulatee;
	pbf_buffer buffer;
	// buffer for physical quantitites query
	//pbf_particle sorted_current_calc;
	//pbf_particle sorted_old_calc;	// one step before
	// neighbor search for fast query
	void storeCurrentPhase();
	// memory allocation utility
	void pbf_particle_alloc(uint32_t num_capacity, dom_dim domain_end);
};

} // end of pbf ns
