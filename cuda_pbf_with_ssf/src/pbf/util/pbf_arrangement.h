#pragma once
#include "../pbf_type.h"
#include <glm\ext.hpp>
#include <cmath>

namespace pbf {

inline void cartesian_line_inclusive(dom_dim_vec& pos, dom_dim origin, dom_dim end, scalar_t space)
{
	auto line_length = glm::distance(origin, end);
	auto line_direction = end - origin;
	line_direction = glm::normalize(line_direction) * space;	// normalized to space
	int particle_num = static_cast<int>(floor(line_length / space)) + 1;
	for (auto i = 0; i < particle_num; ++i) {
		auto coord = origin + line_direction * static_cast<scalar_t>(i);
		pos.push_back(coord);
	}
}

inline uint32_t cartesian_line_inclusive(dom_dim* pos, dom_dim origin, dom_dim end, scalar_t space, uint32_t current_num)
{
	auto line_length = glm::distance(origin, end);
	auto line_direction = end - origin;
	line_direction = glm::normalize(line_direction) * space;	// normalized to space
	uint32_t particle_num = static_cast<uint32_t>(floor(line_length / space)) + 1;
	for (auto i = 0; i < particle_num; ++i) {
		auto coord = origin + line_direction * static_cast<scalar_t>(i);
		pos[current_num + i] = coord;
	}
	return particle_num + current_num;
}

inline void cartesian_line_exclusive(dom_dim_vec& pos, dom_dim origin, dom_dim end, scalar_t space)
{
	auto line_length = glm::distance(origin, end);
	auto line_direction = end - origin;
	line_direction = glm::normalize(line_direction) * space;	// normalized to space
	auto exclusive_origin = origin + line_direction;
	auto coord = exclusive_origin;
	while (true) {
		if (glm::distance(coord - line_direction, end) <= glm::distance(coord - line_direction, coord))
			break;
		pos.push_back(coord);
		coord += line_direction;
	}
}

inline uint32_t cartesian_line_exclusive(dom_dim* pos, dom_dim origin, dom_dim end, scalar_t space, uint32_t current_num)
{
	auto line_length = glm::distance(origin, end);
	auto line_direction = end - origin;
	line_direction = glm::normalize(line_direction) * space;	// normalized to space
	auto exclusive_origin = origin + line_direction;
	auto coord = exclusive_origin;
	uint32_t num = 0;
	while (true) {
		if (glm::distance(coord - line_direction, end) <= glm::distance(coord - line_direction, coord))
			break;
		pos[current_num + num] = coord;
		num++;
		coord += line_direction;
	}
	return num + current_num;
}

inline void subdivide(dom_dim_vec& sub_pos, const dom_dim_vec& pos, scalar_t base_space, scalar_t overlap_space, uint32_t space_ratio)
{
	sub_pos.clear();
	for (auto itr = pos.begin(); itr != pos.end(); ++itr) {
		auto sub_sample_origin = *itr - dom_dim(base_space * 0.5) + dom_dim(overlap_space * 0.5);
		for (int i = 0; i < space_ratio; ++i) {
			for (int j = 0; j < space_ratio; ++j) {
				for (int k = 0; k < space_ratio; ++k) {
					sub_pos.push_back(sub_sample_origin +
						dom_dim(overlap_space * static_cast<double>(i), overlap_space * static_cast<double>(j), overlap_space * static_cast<double>(k)));
				}
			}
		}
	}
}

inline void cartesian_volume(dom_dim_vec& pos, dom_dim origin, dom_dim end, scalar_t space)
{
	auto epsilon = std::numeric_limits<double>::epsilon();
	auto range_xward = end.x - origin.x;
	auto range_yward = end.y - origin.y;
	auto range_zward = end.z - origin.z;
	int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
	int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
	int num_zward = static_cast<int>(range_zward / space + epsilon * 10.0) + 1;
	for (auto z = 0; z < num_zward; ++z){
		for (auto y = 0; y < num_yward; ++y) {
			for (auto x = 0; x < num_xward; ++x) {
				auto coord = origin + dom_dim(space * x, space * y, space * z);
				pos.push_back(coord);
			}
		}
	}
}

inline uint32_t cartesian_volume(dom_dim* pos, dom_dim origin, dom_dim end, scalar_t space, uint32_t current_num)
{
	auto epsilon = std::numeric_limits<double>::epsilon();
	auto range_xward = end.x - origin.x;
	auto range_yward = end.y - origin.y;
	auto range_zward = end.z - origin.z;
	int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
	int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
	int num_zward = static_cast<int>(range_zward / space + epsilon * 10.0) + 1;
	uint32_t num = 0;
	for (auto z = 0; z < num_zward; ++z){
		for (auto y = 0; y < num_yward; ++y) {
			for (auto x = 0; x < num_xward; ++x) {
				auto coord = origin + dom_dim(space * x, space * y, space * z);
				pos[num + current_num] = coord;
				num++;
			}
		}
	}
	return num + current_num;
}

inline void cartesian_rectangle(dom_dim_vec& pos, std::string& axis, scalar_t axis_value, glm::vec2 origin, glm::vec2 end, scalar_t space)
{
	using namespace std;
	auto epsilon = std::numeric_limits<double>::epsilon();
	if (axis == std::string("x")) {
		auto range_yward = end.x - origin.x;
		auto range_zward = end.y - origin.y;
		int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
		int num_zward = static_cast<int>(range_zward / space + epsilon * 10.0) + 1;
		for (auto z = 0; z < num_zward; ++z){
			for (auto y = 0; y < num_yward; ++y) {
				auto coord = dom_dim(axis_value, origin.x, origin.y) + dom_dim(0.0, space * y, space * z);
				pos.push_back(coord);
			}
		}
	}
	else if (axis == std::string("y")) {
		auto range_zward = end.x - origin.x;
		auto range_xward = end.y - origin.y;
		int num_zward = static_cast<int>(range_zward / space + epsilon * 10.0) + 1;
		int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
		for (auto z = 0; z < num_zward; ++z){
			for (auto x = 0; x < num_xward; ++x) {
				auto coord = dom_dim(origin.y, axis_value, origin.x) + dom_dim(space * x, 0.0, space * z);
				pos.push_back(coord);
			}
		}
	}
	else if (axis == std::string("z")) {
		auto range_xward = end.x - origin.x;
		auto range_yward = end.y - origin.y;
		int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
		int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
		for (auto x = 0; x < num_xward; ++x){
			for (auto y = 0; y < num_yward; ++y) {
				auto coord = dom_dim(origin.x, origin.y, axis_value) + dom_dim(space * x, space * y, 0.0);
				pos.push_back(coord);
			}
		}
	}
}

inline uint32_t cartesian_rectangle(dom_dim* pos, std::string& axis, scalar_t axis_value, glm::vec2 origin, glm::vec2 end, scalar_t space, uint32_t current_num)
{
	using namespace std;
	auto epsilon = std::numeric_limits<double>::epsilon();
	uint32_t num = 0;
	if (axis == std::string("x")) {
		auto range_yward = end.x - origin.x;
		auto range_zward = end.y - origin.y;
		int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
		int num_zward = static_cast<int>(range_zward / space + epsilon * 10.0) + 1;
		for (auto z = 0; z < num_zward; ++z){
			for (auto y = 0; y < num_yward; ++y) {
				auto coord = dom_dim(axis_value, origin.x, origin.y) + dom_dim(0.0, space * y, space * z);
				pos[num + current_num] = coord;
				num++;
			}
		}
	}
	else if (axis == std::string("y")) {
		auto range_zward = end.x - origin.x;
		auto range_xward = end.y - origin.y;
		int num_zward = static_cast<int>(range_zward / space + epsilon * 10.0) + 1;
		int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
		for (auto z = 0; z < num_zward; ++z){
			for (auto x = 0; x < num_xward; ++x) {
				auto coord = dom_dim(origin.y, axis_value, origin.x) + dom_dim(space * x, 0.0, space * z);
				pos[num + current_num] = coord;
				num++;
			}
		}
	}
	else if (axis == std::string("z")) {
		auto range_xward = end.x - origin.x;
		auto range_yward = end.y - origin.y;
		int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
		int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
		for (auto x = 0; x < num_xward; ++x){
			for (auto y = 0; y < num_yward; ++y) {
				auto coord = dom_dim(origin.x, origin.y, axis_value) + dom_dim(space * x, space * y, 0.0);
				pos[num + current_num] = coord;
				num++;
			}
		}
	}
	return num + current_num;
}

inline void cartesian_cylinder(dom_dim_vec& pos, std::string& axis, scalar_t axis_value, glm::vec2 origin, scalar_t radius, scalar_t height, scalar_t space)
{
	scalar_t theta = 2.0 * asin(space / (2.0 * radius));
	scalar_t n = floor(2.0 * glm::pi<scalar_t>() / theta);
	scalar_t theta_hat = 2.0 * glm::pi<scalar_t>() / n;

	using namespace std;
	auto epsilon = std::numeric_limits<double>::epsilon();
	if (axis == std::string("x")) {
		for (scalar_t h = 0.0; h < height; h += space) {
			for (scalar_t i = 0.0; i < n; i += 1.0) {
				auto coord = dom_dim(axis_value, origin.x, origin.y) + dom_dim(h, radius * cos(i*theta_hat), radius * sin(i*theta_hat));
				pos.push_back(coord);
			}
		}
	}
	if (axis == std::string("y")) {
		for (scalar_t h = 0.0; h < height; h += space) {
			for (scalar_t i = 0.0; i < n; i += 1.0) {
				auto coord = dom_dim(origin.y, axis_value, origin.x) + dom_dim(radius * sin(i*theta_hat), h, radius * cos(i*theta_hat));
				pos.push_back(coord);
			}
		}
	}
	if (axis == std::string("z")) {
		for (scalar_t h = 0.0; h < height; h += space) {
			for (scalar_t i = 0.0; i < n; i += 1.0) {
				auto coord = dom_dim(origin.x, origin.y, axis_value) + dom_dim(radius * cos(i*theta_hat), radius * sin(i*theta_hat), h);
				pos.push_back(coord);
			}
		}
	}
}


#ifdef SPH_2D
inline void subdivide(dom_dim_vec& sub_pos, const dom_dim_vec& pos, scalar_t base_space, scalar_t overlap_space, uint32_t space_ratio)
{
	sub_pos.clear();
	for (auto itr = pos.begin(); itr != pos.end(); ++itr) {
		auto sub_sample_origin = *itr - glm::vec2(base_space * 0.5, base_space * 0.5) + glm::vec2(overlap_space * 0.5, overlap_space * 0.5);
		for (int i = 0; i < space_ratio; ++i) {
			for (int j = 0; j < space_ratio; ++j) {
				sub_pos.push_back(sub_sample_origin + glm::vec2(overlap_space * static_cast<double>(i), overlap_space * static_cast<double>(j)));
			}
		}
	}
}

inline void cartesian_rectangle(dom_dim_vec& pos, dom_dim origin, dom_dim end, scalar_t space)
{
	auto epsilon = std::numeric_limits<double>::epsilon();
	auto range_xward = end.x - origin.x;
	auto range_yward = end.y - origin.y;
	int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
	int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
	for (auto y = 0; y < num_yward; ++y) {
		for (auto x = 0; x < num_xward; ++x) {
			auto coord = origin + dom_dim(space * x, space * y);
			pos.push_back(coord);
		}
	}
}

inline uint32_t cartesian_rectangle(dom_dim* pos, dom_dim origin, dom_dim end, scalar_t space, uint32_t current_num)
{
	auto epsilon = std::numeric_limits<double>::epsilon();
	auto range_xward = end.x - origin.x;
	auto range_yward = end.y - origin.y;
	int num_xward = static_cast<int>(range_xward / space + epsilon * 10.0) + 1;
	int num_yward = static_cast<int>(range_yward / space + epsilon * 10.0) + 1;
	uint32_t num = 0;
	for (auto y = 0; y < num_yward; ++y) {
		for (auto x = 0; x < num_xward; ++x) {
			auto coord = origin + dom_dim(space * x, space * y);
			pos[num + current_num] = coord;
			num++;
		}
	}
	return num + current_num;
}

#endif

} // end of pbf ns
