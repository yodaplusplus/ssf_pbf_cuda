#pragma once
#include "../common/common_core.h"
#include <map>

class swShaders {
public:
	static swShaders& getInstance() {
		static swShaders instance;
		return instance;
	}
	void enroll(const std::string& key, GLuint shader_id) {
		ShaderName val;
		val.m_shader_name = shader_id;
#ifndef NDEBUG
		val.m_debug_name = key;
#endif
		m_shaders[key] = val;
	}
	void dismiss() {
		const auto itr_end = m_shaders.end();
		for (auto itr = m_shaders.begin(); itr != itr_end; ++itr) {
			const auto shader_name = itr->second.m_shader_name;
			glDeleteShader(shader_name);
		}
	}
	GLuint find(const std::string& key) {
		const auto val = m_shaders[key];
		return val.m_shader_name;
	}

private:
	struct ShaderName {
		GLuint m_shader_name;
#ifndef NDEBUG
		std::string m_debug_name;
#endif
	};
	std::map<std::string, ShaderName> m_shaders;

	swShaders(){}
	swShaders(const swShaders &other);
	swShaders &operator=(const swShaders &other);

};
