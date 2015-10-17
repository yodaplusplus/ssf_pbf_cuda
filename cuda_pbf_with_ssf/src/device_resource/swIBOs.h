#pragma once
#include "../common/common_core.h"
#include <map>

class swIBOs {
public:
	static swIBOs& getInstance() {
		static swIBOs instance;
		return instance;
	}
	void enroll(const std::string& key) {
		GLuint ibo_name;
		glGenBuffers(1, &ibo_name);

		IBOName val;
		val.m_ibo_name = ibo_name;
#ifndef NDEBUG
		val.m_debug_name = key;
#endif
		m_ibos[key] = val;
	}
	void dismiss() {
		const auto itr_end = m_ibos.end();
		for (auto itr = m_ibos.begin(); itr != itr_end; ++itr) {
			const auto ibo_name = itr->second.m_ibo_name;
			glDeleteBuffers(1, &ibo_name);
		}
	}
	GLuint findIBO(const std::string& key) {
		const auto val = m_ibos[key];
		return val.m_ibo_name;
	}
	
private:
	struct IBOName {
		GLuint m_ibo_name;
#ifndef NDEBUG
		std::string m_debug_name;
#endif
	};
	std::map<std::string, IBOName> m_ibos;

	swIBOs(){}
	swIBOs(const swIBOs &other);
	swIBOs &operator=(const swIBOs &other);

};
