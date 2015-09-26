#pragma once
#include "../common/common_core.h"
#include <map>

class swTextures {
public:
	static swTextures& getInstance() {
		static swTextures instance;
		return instance;
	}
	void enroll(const std::string& key) {
		GLuint tex_name;
		glGenTextures(1, &tex_name);

		TexName val;
		val.m_tex_name = tex_name;
#ifndef NDEBUG
		val.m_debug_name = key;
#endif
		m_Textures[key] = val;
	}
	void dismiss() {
		const auto itr_end = m_Textures.end();
		for (auto itr = m_Textures.begin(); itr != itr_end; ++itr) {
			const auto tex_name = itr->second.m_tex_name;
			glDeleteTextures(1, &tex_name);
		}
	}
	GLuint find(const std::string& key) {
		const auto val = m_Textures[key];
		return val.m_tex_name;
	}

private:
	struct TexName {
		GLuint m_tex_name;
#ifndef NDEBUG
		std::string m_debug_name;
#endif
	};
	std::map<std::string, TexName> m_Textures;

	swTextures(){}
	swTextures(const swTextures &other);
	swTextures &operator=(const swTextures &other);

};
