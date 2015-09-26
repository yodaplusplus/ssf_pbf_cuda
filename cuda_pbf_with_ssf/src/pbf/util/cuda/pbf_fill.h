#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda {

void fill(uint32_t* target, uint32_t value, uint32_t num);

void fill(dom_dim* target, dom_dim value, uint32_t num);

void fill(scalar_t* target, scalar_t value, uint32_t num);

}	// end of cuda ns
}	// end of pbf ns
