#version 450

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
	mat4x4 InvProjection;
};

uniform mat4x4 World;
uniform sampler2D tex_depth;
uniform sampler2D tex_normal;
//uniform sampler2D u_Positiontex;
//uniform sampler2D u_Thicktex;

in block {
  vec3 position;
  vec2 tex_coord;
} IN;

out vec4 out_Color;

//Depth used in the Z buffer is not linearly related to distance from camera
//This restores linear depth
float linearizeDepth(float exp_depth, float near, float far) {
    return	(2 * near) / (far + near -  exp_depth * (far - near));
}

vec3 uvToEye(vec2 texCoord, float depth){
	float x = texCoord.x * 2.0 - 1.0;
	float y = texCoord.y * -2.0 + 1.0;
	vec4 clipPos = vec4(x , y, depth, 1.0f);
	vec4 viewPos = InvProjection * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main()
{
	//Uniform Light Direction (Billboard)
  vec4 lightDir = View * World * vec4(1.f, 4.0f, 1.0f, 0.0f);

  //Get Texture Information about the Pixel
  vec3 N = texture(tex_normal, IN.tex_coord).xyz;
  float exp_depth = texture(tex_depth, IN.tex_coord).r;
  // float lin_depth = linearizeDepth(exp_depth,u_Near,u_Far);
	//float thickness = clamp(texture(u_Thicktex,fs_Texcoord).r,0.0f,1.0f);
  vec3 position = uvToEye(IN.tex_coord, exp_depth).xyz;

	if(exp_depth == 1.0) {
		discard;
	}

  vec3 incident = normalize(lightDir.xyz);
  vec3 viewer = normalize(-position.xyz);

	vec3 Color = vec3(0.0, 0.0, 1.0);

  //Blinn-Phong Shading Coefficients
  vec3 H = normalize(incident + viewer);
  float specular = pow(max(0.0f, dot(H,N)),50.0f);
  float diffuse = max(0.0f, dot(incident, N));
  //Fresnel Reflection
  float r_0 = 0.3f;
  float fres_refl = r_0 + (1-r_0)*pow(1-dot(N,viewer),5.0f);

	//Color Attenuation from Thickness
  //(Beer's Law)
  //float k_r = 5.0f;
  //float k_g = 1.0f;
  //float k_b = 0.1f;
  //vec3 color_atten = vec3( exp(-k_r*thickness), exp(-k_g*thickness), exp(-k_b*thickness));

  //Final Real Color Mix
  //float transparency = 1-thickness;
  //vec3 final_color = mix(color_atten.rgb * diffuse, refrac_color.rgb,transparency);

	gl_FragDepth = exp_depth;
	out_Color = vec4(Color.rgb * diffuse + specular * vec3(1.0f), 1.0f) + vec4(0.05f, 0.05f, 0.08f, 0.f);
	//out_Color = vec4(final_color.rgb + specular * vec3(1.0f) + refl_color.rgb * fres_refl, 1.0f);

	return;
}
