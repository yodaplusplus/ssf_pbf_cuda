#pragma once
#include "../../pbf_type.h"

namespace pbf {

template<typename T> class ContributionTraits;

namespace cuda {
namespace detail {

template<typename func_t, typename arg_t>
__host__ __device__ typename ContributionTraits<func_t>::ConT calcContribution(
	dom_dim self_pos,
	const uint32_t* cell_start,
	const uint32_t* cell_end,
	scalar_t cell_width,
	dom_udim grid_size,
	func_t& contribute,
	arg_t& arg_list);

} // end of detail ns
} // end of cuda ns
} // end of pbf ns

#include "detail/pbf_contribution.inl"
