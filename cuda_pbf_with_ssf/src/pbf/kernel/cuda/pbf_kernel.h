#pragma once
#include "../../pbf_type.h"
#include "detail/pbf_kernel.inl"

namespace pbf {
namespace kernel {
namespace cuda {
;
template<typename kernel_t>
__host__ __device__
inline scalar_t weight(scalar_t r, scalar_t inv_h)
{
	return kernel_t::original(r, inv_h);
}

template<typename kernel_t>
__host__ __device__
inline scalar_t weight_deriv(scalar_t r, scalar_t inv_h)
{
	return kernel_t::derivative(r, inv_h);
}

CLASS_KERNEL(PBFKERNEL, detail::Wpoly6, detail::WspikyDerivative)

} // end of cuda ns
} // end of kernel ns
} // end of pbf ns
