// builtin arrangement of uniform buffer object
#pragma once
#include "../common_core.h"

namespace UBO {
struct Scene {
	glm::mat4x4 proj;
	glm::mat4x4 view;
	glm::mat4x4 inv_proj;
};

}




