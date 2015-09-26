#pragma once
#include "../common/common_core.h"
#include <map>

class swVBOs {
public:
	static swVBOs& getInstance() {
		static swVBOs instance;
		return instance;
	}
	void enroll(const std::string& key) {
		GLuint vbo_name;
		glGenBuffers(1, &vbo_name);

		VBOName val;
		val.m_vbo_name = vbo_name;
#ifndef NDEBUG
		val.m_debug_name = key;
#endif
		m_vbos[key] = val;
	}
	void dismiss() {
		const auto itr_end = m_vbos.end();
		for (auto itr = m_vbos.begin(); itr != itr_end; ++itr) {
			const auto vbo_name = itr->second.m_vbo_name;
			glDeleteBuffers(1, &vbo_name);
		}
	}
	GLuint findVBO(const std::string& key) {
		const auto val = m_vbos[key];
		return val.m_vbo_name;
	}

private:
	struct VBOName {
		GLuint m_vbo_name;
#ifndef NDEBUG
		std::string m_debug_name;
#endif
	};
	std::map<std::string, VBOName> m_vbos;

	swVBOs(){}
	swVBOs(const swVBOs &other);
	swVBOs &operator=(const swVBOs &other);

};
