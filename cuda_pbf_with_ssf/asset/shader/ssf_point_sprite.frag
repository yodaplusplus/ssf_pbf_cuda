#version 330

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
  mat4x4 InvProjection;
};

uniform float pointRadius;  // point size in world space

in block {
	vec3 pos_from_eye;
};

out vec4 frag_color;

void main(void)
{
    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

    //calculate depth
    vec4 pixelPos = vec4(pos_from_eye + N*pointRadius, 1.0f);
    vec4 clipSpacePos = Projection * pixelPos;
    gl_FragDepth = clipSpacePos.z / clipSpacePos.w;
    frag_color = vec4(N, 1.0);
}
