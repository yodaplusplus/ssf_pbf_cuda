#pragma once
#include "../common/common_core.h"
#include <map>

class swVAOs {
public:
	static swVAOs& getInstance() {
		static swVAOs instance;
		return instance;
	}
	void enroll(const std::string& key) {
		GLuint vao_name;
		glGenVertexArrays(1, &vao_name);

		VAOName val;
		val.m_vao_name = vao_name;
#ifndef NDEBUG
		val.m_debug_name = key;
#endif
		m_VAOs[key] = val;
	}
	void dismiss() {
		const auto itr_end = m_VAOs.end();
		for (auto itr = m_VAOs.begin(); itr != itr_end; ++itr) {
			const auto vao_name = itr->second.m_vao_name;
			glDeleteVertexArrays(1, &vao_name);
		}
	}
	GLuint find(const std::string& key) {
		const auto val = m_VAOs[key];
		return val.m_vao_name;
	}

private:
	struct VAOName {
		GLuint m_vao_name;
#ifndef NDEBUG
		std::string m_debug_name;
#endif
	};
	std::map<std::string, VAOName> m_VAOs;

	swVAOs(){}
	swVAOs(const swVAOs &other);
	swVAOs &operator=(const swVAOs &other);

};
