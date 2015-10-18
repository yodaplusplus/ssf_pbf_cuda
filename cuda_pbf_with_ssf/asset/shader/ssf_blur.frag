#version 450
#pragma optionNV(fastmath on)
#pragma optionNV(inline all)
#pragma optionNV (unroll all)

layout(std140) uniform Scene {
	mat4x4 Projection;
	mat4x4 View;
	mat4x4 InvProjection;
};

uniform sampler2D tex_depth;

uniform float proj_far;
uniform float proj_near;
uniform float blur_scale;
uniform float blur_radius;

in block {
	vec2 tex_coord;
};

out float out_depth;

//Depth used in the Z buffer is not linearly related to distance from camera
//This restores linear depth
float linearizeDepth(float exp_depth, float near, float far) {

	return	(2 * near) / (far + near -  exp_depth * (far - near));
}

float exponentializeDepth(float lin_depth, float near, float far) {
	return (far + near - (2.0f * near) / lin_depth) / (far - near);
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
	//Get Depth Information about the Pixel
	float exp_depth = texture(tex_depth, tex_coord).r;
	float lin_depth = linearizeDepth(exp_depth, proj_near, proj_far);
	float sum = 0;
	float wsum = 0;

	if(exp_depth == 1.0f){
		out_depth = 1.0f;
		return;
	}

	vec3 position = uvToEye(tex_coord, exp_depth).xyz;
	float dist = length(position);
	float blurRadius = blur_radius * blur_scale / dist;

	const int windowWidth = 4;
	for(int x = -windowWidth; x < windowWidth; x++){
      vec2 sample_coord = vec2(tex_coord.s + x*blurRadius, tex_coord.t);
      float sampleDepth = texture(tex_depth, sample_coord).r;
			float sample_lin_depth = linearizeDepth(sampleDepth, proj_near, proj_far);

      if(sampleDepth != 1.0f){
          //Spatial
          float r = x * x * blurRadius;
          float w = exp(-r);

          //Range
					float r2 = 0.0f;
					float pair_depth = 0.0f;
					float diff = sample_lin_depth - lin_depth;
					if(abs(diff) < 0.01f) {
        		r2 = diff * 100.0f;
					}
					else {
						r2 = 0.f;
					}
          float g = exp(-r2*r2);

          sum += sample_lin_depth * w * g;
          wsum += w * g;
      }
			// else {
			// 	//Spatial
			// 	float r = x * x * blurRadius;
			// 	float w = exp(-r);
			//
			// 	sum += lin_depth * w;
			// 	wsum += w;
			// }
	}
	for(int y = -windowWidth; y < windowWidth; y++){
			vec2 sample_coord = vec2(tex_coord.s, tex_coord.t + y*blurRadius);
			float sampleDepth = texture(tex_depth, sample_coord).r;
			float sample_lin_depth = linearizeDepth(sampleDepth, proj_near, proj_far);

			if(sampleDepth != 1.0f){
					//Spatial
					float r = y * y * blurRadius;
					float w = exp(-r);

					//Range
					float r2 = 0.0f;
					float pair_depth = 0.0f;
					float diff = sample_lin_depth - lin_depth;
					if(abs(diff) < 0.01f) {
						r2 = diff * 100.0f;
					}
					else {
						r2 = 0.f;
					}
					float g = exp(-r2*r2);

					sum += sample_lin_depth * w * g;
					wsum += w * g;
			}
			// else {
			// 	//Spatial
			// 	float r = y * y * blurRadius;
			// 	float w = exp(-r);
			//
			// 	sum += lin_depth * w;
			// 	wsum += w;
			// }
	}

	if(wsum > 0.0f){
		sum = sum/wsum;
	}
	sum = exponentializeDepth(sum, proj_near, proj_far);

	out_depth = sum;
	gl_FragDepth = sum;
	//out_Depth = vec4(exp_depth);
}
