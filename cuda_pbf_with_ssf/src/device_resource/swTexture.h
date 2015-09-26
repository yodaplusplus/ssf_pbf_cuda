#pragma once
#include "../common/common_core.h"
#include <functional>

// for texture2d
class swTexture {
public:
	swTexture() {
		glGenTextures(1, &m_tex_name);
	}
	~swTexture() {
		glDeleteTextures(1, &m_tex_name);
	}
	GLuint getTextureName() {
		return m_tex_name;
	}
	void setTextureDescription(const std::function<void(void)>& f) {
		glBindTexture(GL_TEXTURE_2D, m_tex_name);
		f();
		glBindTexture(GL_TEXTURE_2D, 0);
	}

private:
	GLuint m_tex_name;
	void operator=(const swTexture&);
	swTexture(const swTexture&);
};
