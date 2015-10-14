#pragma once

#define CLASS_KERNEL(class_name, kernel_name_original, kernel_name_deriv)\
class class_name {\
public:\
	__host__ __device__\
	static scalar_t original(scalar_t r, scalar_t h) {\
		return kernel_name_original(r, h);\
		}\
	__host__ __device__\
	static scalar_t derivative(scalar_t r, scalar_t h) {\
		return kernel_name_deriv(r, h);\
		}\
};

namespace pbf {
namespace kernel {
namespace detail {
;
//const scalar_t PI = glm::pi<scalar_t>();
//const scalar_t inv_pi = glm::one_over_pi<scalar_t>();

// Lucy 1977 for density calculation
__host__ __device__ inline scalar_t WLucy1977(scalar_t r, scalar_t inv_h)
{
	const scalar_t inv_pi = 0.3183098861;
	auto q = r * inv_h;
	if (0.0 <= q && q <= 1.0)
		return (105.0 / 16.0 * inv_pi * pow(inv_h, 3)) * (1.0 + 3.0 * q) * pow(1.0 - q, 3);
	else
		return 0.0;
}

__host__ __device__ inline scalar_t Wpoly6(scalar_t r, scalar_t inv_h)
{
	//const scalar_t inv_pi = 0.3183098861;
	auto q = r * inv_h;
	//auto alpha = 315.f / 64.f * inv_pi / (h*h*h);
	//auto alpha = 4.921875f * inv_pi / (h*h*h);
	auto alpha = 1.56668147065f * inv_h * inv_h * inv_h;
	if (q <= 1.f) {
		auto q2 = q * q;
		auto t = 1.f - q2;
		return alpha * t * t * t;
	}
	else
		return 0.f;
}

__host__ __device__ inline scalar_t WspikyDerivative(scalar_t r, scalar_t inv_h)
{
	//const scalar_t inv_pi = 0.3183098861;
	auto q = r * inv_h;
	auto inv_h2 = inv_h * inv_h;
	if (q <= 1.f) {
		auto t = 1.f - q;
		auto inv_h4 = inv_h2 * inv_h2;
		return -14.3239448745f * inv_h4 * t * t;
		//return -45.f * inv_pi / (h*h*h*h) * (1.f - q) * (1.f - q);
	}
	else
		return 0.f;
}

#ifdef PBF_2D
inline scalar_t Wpoly6(scalar_t r, scalar_t h)
{
	auto q = r / h;
	auto alpha = 4.0 * inv_pi / pow(h, 2);
	if (0.0 <= q && q <= 1.0)
		return alpha * pow(1.0 - pow(q, 2), 3);
	else
		return 0.0;
}

inline scalar_t WspikyGradient(scalar_t r, scalar_t h)
{
	auto q = r / h;
	if (0.0 <= q && q <= 1.0)
		return -30.0 * inv_pi / pow(h, 3) * pow(1.0 - q, 2);
	else
		return 0.0;
}
#endif

} // end of detail ns
} // end of kernel ns
} // end of pbf ns
