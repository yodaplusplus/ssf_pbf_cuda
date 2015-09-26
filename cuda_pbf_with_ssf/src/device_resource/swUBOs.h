#pragma once
#include "../common/common_core.h"
#include <map>

class swUBOs {
public:
	static swUBOs& getInstance() {
		static swUBOs instance;
		return instance;
	}
	void enroll(const std::string& key) {
		GLuint ubo_name;
		glGenBuffers(1, &ubo_name);

		UBOName val;
		val.m_ubo_name = ubo_name;
#ifndef NDEBUG
		val.m_debug_name = key;
#endif
		m_UBOs[key] = val;
	}
	void dismiss() {
		const auto itr_end = m_UBOs.end();
		for (auto itr = m_UBOs.begin(); itr != itr_end; ++itr) {
			const auto ubo_name = itr->second.m_ubo_name;
			glDeleteBuffers(1, &ubo_name);
		}
	}
	GLuint find(const std::string& key) {
		const auto val = m_UBOs[key];
		return val.m_ubo_name;
	}

private:
	struct UBOName {
		GLuint m_ubo_name;
#ifndef NDEBUG
		std::string m_debug_name;
#endif
	};
	std::map<std::string, UBOName> m_UBOs;

	swUBOs(){}
	swUBOs(const swUBOs &other);
	swUBOs &operator=(const swUBOs &other);

};
