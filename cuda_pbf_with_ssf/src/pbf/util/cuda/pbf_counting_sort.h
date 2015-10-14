#pragma once
#include "../../pbf_type.h"

namespace pbf {
namespace cuda {

void insertAndCount(
	uint32_t* cell_count, uint32_t* hash_count,
	const uint32_t* hash,
	uint32_t num_item);

void getPrefixSumStorageSize(
	size_t& temp_storage_size,
	uint32_t num_item
	);

void prefixSum(
	void* temp_storage,
	size_t temp_storage_size,
	uint32_t* cumulative_cell_count,
	const uint32_t* cell_count,
	uint32_t num_cell
	);

void prefixSum(
	uint32_t* cell_count,
	uint32_t num_cell
	);

void sortByCounting(
	uint32_t* sorted_hash,
	const uint32_t* hash, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item
	);

void sortByCounting(
	uint32_t* sorted_hash, uint32_t* sorted_index,
	const uint32_t* hash, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item
	);

void countingSort(
	uint32_t* sorted_hash, uint32_t* sorted_index,
	uint32_t* cell_count, uint32_t* hash_count,
	const uint32_t* hash,
	uint32_t num_item, uint32_t num_cell
	);

void countingSort(
	uint32_t* sorted_hash, uint32_t* sorted_index,
	uint32_t* cumulative_cell_count, uint32_t* cell_count, uint32_t* hash_count,
	void* temp_storage, size_t temp_storage_size,	// storage for scan
	const uint32_t* hash,
	uint32_t num_item, uint32_t num_cell
	);

// for z-order hash, 64 bit hashing
// uint32_t* hash -> uint64_t* hash
		
void insertAndCount(
	uint32_t* cell_count, uint32_t* hash_count,
	const uint64_t* hash,
	uint32_t num_item);

void sortByCounting(
	uint64_t* sorted_hash, uint32_t* sorted_index,
	const uint64_t* hash, const uint32_t* cumulative_cell_count, const uint32_t* hash_count,
	uint32_t num_item
	);

void countingSort(
	uint64_t* sorted_hash, uint32_t* sorted_index,
	uint32_t* cumulative_cell_count, uint32_t* cell_count, uint32_t* hash_count,
	void* temp_storage, size_t temp_storage_size,	// storage for scan
	const uint64_t* hash,
	uint32_t num_item, uint32_t num_cell
	);


}	// end of cuda ns
}	// end of pbf ns

