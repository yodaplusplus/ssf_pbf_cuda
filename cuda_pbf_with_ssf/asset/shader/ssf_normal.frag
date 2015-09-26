#version 330

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
	mat4x4 InvProjection;
};

uniform sampler2D tex_depth;

in block {
	vec2 tex_coord;
};

out vec4 out_Normal;

vec3 uvToEye(vec2 texCoord, float depth){
	float x = texCoord.x * 2.0 - 1.0;
	float y = texCoord.y * 2.0 - 1.0;
	vec4 clipPos = vec4(x, y, depth, 1.0f);
	vec4 viewPos = InvProjection * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main()
{
  //Get Depth Information about the Pixel
  float exp_depth = texture(tex_depth, tex_coord).r;
  if(exp_depth == 1.0)
    discard;

	vec3 position = uvToEye(tex_coord, exp_depth);

	//Compute Gradients of Depth and Cross Product Them to Get Normal
	out_Normal = vec4(normalize(cross(dFdx(position.xyz), dFdy(position.xyz))), 1.0f);
}
