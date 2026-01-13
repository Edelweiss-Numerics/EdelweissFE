#ifndef CSR_CORE_H
#define CSR_CORE_H

#include <vector>
#include <algorithm>
#include <execution> // C++17 Parallel algorithms
#include <ranges>    // std::views::iota (C++20)
#include <cstdint>
#include <numeric>
#include <span>      // C++20

// Define Entry
struct Entry {
    int64_t row;
    int64_t col;
    int32_t origIdx;

    // Standard comparison for sorting
    bool operator<(const Entry& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

class CSRCore {
public:
    // --- CSR Outputs ---
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    int64_t nnz = 0;
    int64_t nDof = 0;

    // --- Assembly Acceleration Maps (The "Gather" Structure) ---
    // Replaces 'map_to_csr'.
    // gather_sources: A permutation vector. Maps sorted position -> original V index.
    // assembly_ptr:   pointers into gather_sources. Entry 'i' sums V values from
    //                 gather_sources[assembly_ptr[i]] to gather_sources[assembly_ptr[i+1]]
    std::vector<int32_t> gather_sources;
    std::vector<int32_t> assembly_ptr;

    CSRCore(long* I, long* J, int64_t n_pairs, int64_t n_dof) : nDof(n_dof) {

        // --- CONFIGURATION ---
        const int64_t NUM_PARTITIONS = 2048;
        double chunk_size = (double)nDof / NUM_PARTITIONS;

        // --- STEP 1: HISTOGRAM ---
        std::vector<int32_t> p_counts(NUM_PARTITIONS, 0);
        for (int64_t k = 0; k < n_pairs; ++k) {
            int p_id = (int)(I[k] / chunk_size);
            if (p_id >= NUM_PARTITIONS) p_id = NUM_PARTITIONS - 1;
            p_counts[p_id]++;
        }

        // --- STEP 2: OFFSETS ---
        std::vector<int32_t> p_offsets(NUM_PARTITIONS + 1, 0);
        int32_t current_offset = 0;
        for (int i = 0; i < NUM_PARTITIONS; ++i) {
            p_offsets[i] = current_offset;
            current_offset += p_counts[i];
        }
        p_offsets[NUM_PARTITIONS] = current_offset;

        // --- STEP 3: FLAT SCATTER ---
        std::vector<Entry> flat_buffer(n_pairs);
        std::vector<int32_t> heads = p_offsets;

        for (int64_t k = 0; k < n_pairs; ++k) {
            int64_t r = I[k];
            int p_id = (int)(r / chunk_size);
            if (p_id >= NUM_PARTITIONS) p_id = NUM_PARTITIONS - 1;

            int32_t pos = heads[p_id]++;
            flat_buffer[pos] = {r, J[k], (int32_t)k};
        }

        // --- STEP 4: PARALLEL SORT OF CHUNKS ---
        std::vector<int> p_ids(NUM_PARTITIONS);
        std::iota(p_ids.begin(), p_ids.end(), 0);

        std::for_each(std::execution::par_unseq, p_ids.begin(), p_ids.end(),
            [&](int p) {
                int32_t start = p_offsets[p];
                int32_t end = p_offsets[p+1];
                if (start < end) {
                    std::sort(flat_buffer.begin() + start, flat_buffer.begin() + end);
                }
            }
        );

        // --- STEP 5: BUILD CSR & GATHER MAPS (Serial Scan) ---

        // Pre-allocate worst case
        indices.reserve(n_pairs);
        indptr.assign(nDof + 1, 0);

        // Gather structures
        gather_sources.resize(n_pairs);     // Stores the permutation
        assembly_ptr.reserve(n_pairs + 1);  // Stores the ranges
        assembly_ptr.push_back(0);

        if (n_pairs > 0) {
            // Handle first element
            const auto& first = flat_buffer[0];
            indices.push_back((int32_t)first.col);
            gather_sources[0] = first.origIdx;

            // Row counting
            std::vector<int32_t> row_counts(nDof, 0);
            row_counts[first.row]++;

            for (int64_t k = 1; k < n_pairs; ++k) {
                const auto& curr = flat_buffer[k];
                const auto& prev = flat_buffer[k-1];

                // Save the source index for the gather map
                gather_sources[k] = curr.origIdx;

                bool is_new_entry = (curr.row != prev.row) || (curr.col != prev.col);

                if (is_new_entry) {
                    // Previous entry sequence ended at k
                    assembly_ptr.push_back(k);

                    // Start new CSR entry
                    indices.push_back((int32_t)curr.col);
                    row_counts[curr.row]++;
                }
                // If not new, we just continue (the 'k' is effectively added to the current range)
            }
            // Close the last range
            assembly_ptr.push_back(n_pairs);

            this->nnz = indices.size();

            // Finalize Indptr
            int32_t acc = 0;
            for(int64_t r=0; r<nDof; ++r) {
                indptr[r] = acc;
                acc += row_counts[r];
            }
            indptr[nDof] = acc;
        }
    }

    void update(const double* V_data, double* csr_data) const {

        // Iterate over the Non-Zeros (the output array)
        auto range = std::views::iota((int64_t)0, nnz);

        std::for_each(std::execution::par_unseq, range.begin(), range.end(),
            [this, V_data, csr_data](int64_t i) {

                // Identify the range of input values that sum to this matrix entry
                int32_t start = assembly_ptr[i];
                int32_t end   = assembly_ptr[i+1];

                double sum = 0.0;

                // Vectorizable summation loop (no atomics needed!)
                // The CPU can pipeline these loads efficiently.
                #pragma omp simd reduction(+:sum) // Hint for compiler (optional)
                for (int32_t k = start; k < end; ++k) {
                    // Random Read from V (Cache friendly: Read-Only Shared state)
                    sum += V_data[gather_sources[k]];
                }

                // Single Write (Cache friendly: Exclusive state, no contention)
                csr_data[i] = sum;
            }
        );
    }
};

#endif
