#version 330

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
	mat4x4 InvProjection;
};

uniform mat4x4 World;
uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

in vec3 Position;

out block {
	vec3 pos_from_eye;
};

void main(void) {
	mat4x4 vw = View * World;
	vec3 posEye = (vw * vec4(Position.xyz, 1.0f)).xyz;
	float dist = length(posEye);
	gl_PointSize = pointRadius * (pointScale/dist);

	pos_from_eye = posEye;
	gl_Position = Projection * vec4(posEye, 1.0);
}
