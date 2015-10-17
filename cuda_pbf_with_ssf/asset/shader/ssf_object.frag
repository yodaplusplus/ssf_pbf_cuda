#version 450

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
  mat4x4 InvProjection;
};

uniform float point_radius;

in block {
  vec2 vertex_uv;
  vec3 pos_from_eye;
};

// out vec4 frag_color;

void main (void)
{
  // frag_color = vec4(0.0, 0.0, 1.0, 1.0);
  // return;

  // calculate normal from texture coordinates
  vec3 N;
  N.xy = vertex_uv.xy * vec2(2.0, 2.0) - vec2(1.0, 1.0);
  float mag = dot(N.xy, N.xy);
  if (mag > 1.0) discard;   // kill pixels outside circle
  N.z = sqrt(1.0-mag);

  //calculate depth
  vec4 pixelPos = vec4(pos_from_eye + N * point_radius, 1.0f);
  vec4 clipSpacePos = Projection * pixelPos;
  gl_FragDepth = clipSpacePos.z / clipSpacePos.w;
  // frag_color = vec4(N, 1.0);
}
