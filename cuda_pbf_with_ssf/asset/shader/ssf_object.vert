#version 330

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
	mat4x4 InvProjection;
};

uniform mat4x4 World;

in vec3 Position;

void main(void)
{
  mat4x4 vw = View * World;
  gl_Position = vw * vec4(Position, 1.0);
}
