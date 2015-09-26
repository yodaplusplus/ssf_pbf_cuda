#version 330

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
	mat4x4 InvProjection;
};

uniform float point_radius;

out block {
  vec2 vertex_uv;
  vec3 pos_from_eye;
};

void main (void)
{
  vec4 P = gl_in[0].gl_Position;

  // a: left-bottom
  vec2 va = P.xy + vec2(-0.5, -0.5) * point_radius;
  gl_Position = Projection * vec4(va, P.zw);
  vertex_uv = vec2(0.0, 0.0);
  pos_from_eye = vec3(va, P.z);
  EmitVertex();

  // d: right-bottom
  vec2 vd = P.xy + vec2(0.5, -0.5) * point_radius;
  gl_Position = Projection * vec4(vd, P.zw);
  vertex_uv = vec2(1.0, 0.0);
  pos_from_eye = vec3(vd, P.z);
  EmitVertex();

  // b: left-top
  vec2 vb = P.xy + vec2(-0.5, 0.5) * point_radius;
  gl_Position = Projection * vec4(vb, P.zw);
  vertex_uv = vec2(0.0, 1.0);
  pos_from_eye = vec3(vb, P.z);
  EmitVertex();

  // c: right-top
  vec2 vc = P.xy + vec2(0.5, 0.5) * point_radius;
  gl_Position = Projection * vec4(vc, P.zw);
  vertex_uv = vec2(1.0, 1.0);
  pos_from_eye = vec3(vc, P.z);
  EmitVertex();

  EndPrimitive();
}
