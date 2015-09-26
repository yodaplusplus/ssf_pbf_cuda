#pragma once
#include "../common/common_core.h"
#include <map>

class swFBOs {
public:
	static swFBOs& getInstance() {
		static swFBOs instance;
		return instance;
	}
	void enroll(const std::string& key) {
		GLuint fbo_name;
		glGenFramebuffers(1, &fbo_name);

		FBOName val;
		val.m_fbo_name = fbo_name;
#ifndef NDEBUG
		val.m_debug_name = key;
#endif
		m_FBOs[key] = val;
	}
	void dismiss() {
		const auto itr_end = m_FBOs.end();
		for (auto itr = m_FBOs.begin(); itr != itr_end; ++itr) {
			const auto fbo_name = itr->second.m_fbo_name;
			glDeleteFramebuffers(1, &fbo_name);
		}
	}
	GLuint find(const std::string& key) {
		const auto val = m_FBOs[key];
		return val.m_fbo_name;
	}

private:
	struct FBOName {
		GLuint m_fbo_name;
#ifndef NDEBUG
		std::string m_debug_name;
#endif
	};
	std::map<std::string, FBOName> m_FBOs;

	swFBOs(){}
	swFBOs(const swFBOs &other);
	swFBOs &operator=(const swFBOs &other);

};
