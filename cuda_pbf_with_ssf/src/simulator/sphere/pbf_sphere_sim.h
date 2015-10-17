#pragma once
#include "../pbf_base_sim.h"
#include "pbf_sphere_init_cond.h"

namespace pbf {
;
class pbf_sphere_sim : public pbf_base_sim
{
public:
	pbf_sphere_sim(scalar_t space);
	~pbf_sphere_sim();
	void simulateOneStep();
	uint32_t getParticlesNum();
	void getParticlesPosition(std::vector<glm::vec3>& particle);
	GLuint getParticlesPositionVBO();
	void drawNeighborSearchArea(GLuint ubo_scene_id);

private:
	std::shared_ptr<pbf_sphere_init_cond> init_cond;
	cudaGraphicsResource_t cu_res;
	class jet;
	std::shared_ptr<jet> m_jet;
	// grid drawing for debug
	class neighbor_search_area;
	class area_drawing;
	std::shared_ptr<area_drawing> m_nsa;
	std::shared_ptr<area_drawing> m_sim_area;
};

} // end of pbf ns
