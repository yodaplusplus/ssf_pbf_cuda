#pragma once
#include "../pbf_base_sim.h"
#include "pbf_dam_init_cond.h"

namespace pbf {
;
class pbf_dam_sim : public pbf_base_sim
{
public:
	pbf_dam_sim(scalar_t space);
	~pbf_dam_sim();
	void simulateOneStep();
	uint32_t getParticlesNum();
	void getParticlesPosition(std::vector<glm::vec3>& particle);
	GLuint getParticlesPositionVBO();

private:
	std::shared_ptr<pbf_dam_init_cond> init_cond;
	cudaGraphicsResource_t cu_res;
};

} // end of pbf ns
