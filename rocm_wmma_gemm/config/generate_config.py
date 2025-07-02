#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import defaultdict

def generate_config_header(config_file, output_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract unique configurations
    unique_configs = set()
    for conf in config['configurations']:
        cfg = conf['config']
        unique_configs.add((cfg['warps_m'], cfg['warps_n'],
                            cfg['warp_tile_m'], cfg['warp_tile_n']))

    # Always add default config if not present
    default_config = (4, 4, 4, 4)
    if default_config not in unique_configs:
        unique_configs.add(default_config)

    # Sort configs for stable output and indexing
    unique_configs = sorted(list(unique_configs))
    num_configs = len(unique_configs)

    # Group configurations by matrix dimensions and (A,B) layout
    size_ab_configs = {}
    for conf in config['configurations']:
        range_info = conf['range']
        M, N, K = range_info['M'], range_info['N'], range_info['K']

        # Create key for this matrix size
        size_key = (M, N, K)

        # Create layout key from (A,B) layout only - C layout doesn't matter
        layout_dict = conf['layout']
        ab_layout_key = (
            layout_dict.get('A', 'any'),
            layout_dict.get('B', 'any')
        )

        if size_key not in size_ab_configs:
            size_ab_configs[size_key] = {}

        cfg = conf['config']
        config_tuple = (cfg['warps_m'], cfg['warps_n'],
                        cfg['warp_tile_m'], cfg['warp_tile_n'])
        config_idx = unique_configs.index(config_tuple)

        size_ab_configs[size_key][ab_layout_key] = config_idx

    # Sort sizes for search
    sorted_sizes = sorted(size_ab_configs.keys())

    # Also create a flattened list of all configs for fallback search
    all_configs = []
    for size_key, ab_layouts in size_ab_configs.items():
        M, N, K = size_key
        for ab_layout_key, config_idx in ab_layouts.items():
            all_configs.append({
                'size': (M, N, K),
                'ab_layout': ab_layout_key,
                'config_idx': config_idx
            })

    # Generate code
    code = f"""// Auto-generated file - DO NOT EDIT
#ifndef ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP
#define ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP

#include <rocm_wmma_gemm/kernel/common.hpp>
#include <array>
#include <tuple>
#include <cstddef>
#include <cmath>
#include <limits>

namespace rocm_wmma_gemm
{{

// Number of unique kernel variants
static constexpr size_t KERNEL_VARIANTS = {num_configs};

// Configuration parameters for a specific problem size
struct gemm_params
{{
    int warps_m;
    int warps_n;
    int warp_tile_m;
    int warp_tile_n;
}};

namespace detail
{{
    // Kernel configuration tuple
    using kernel_config = std::tuple<int, int, int, int>; // warps_m, warps_n, warp_tile_m, warp_tile_n

    // All unique kernel configurations
    static constexpr std::array<kernel_config, KERNEL_VARIANTS> kernel_configs = {{
"""

    # Generate config array
    for i, (wm, wn, wtm, wtn) in enumerate(unique_configs):
        code += f"        std::tuple<int, int, int, int>{{{wm}, {wn}, {wtm}, {wtn}}}"
        code += "," if i < len(unique_configs) - 1 else ""
        code += f" // Config {i}\n"

    code += """    };

    // Default config (last in the array)
    static constexpr size_t DEFAULT_CONFIG_IDX = KERNEL_VARIANTS - 1;

    // For finding closest configuration when exact match not found
    struct size_ab_config_entry
    {
        size_t m, n, k;
        m_layout layout_a;
        m_layout layout_b;
        size_t config_idx;
    };

    static constexpr std::array<size_ab_config_entry, """ + str(len(all_configs)) + """> all_configs = {{
"""

    # Generate flattened config array for fallback search
    for i, entry in enumerate(all_configs):
        M, N, K = entry['size']
        a_layout, b_layout = entry['ab_layout']
        config_idx = entry['config_idx']

        # Convert layout strings to enum values
        a_enum = f"m_layout::{a_layout}" if a_layout != "any" else "m_layout::row_major"
        b_enum = f"m_layout::{b_layout}" if b_layout != "any" else "m_layout::row_major"

        code += f"        {{{M}, {N}, {K}, {a_enum}, {b_enum}, {config_idx}}}"
        code += "," if i < len(all_configs) - 1 else ""
        code += "\n"

    code += """    }};

    // Find closest configuration when exact match not found
    // This is a fallback mechanism only used when switch-case doesn't find a match
    constexpr size_t find_closest_config(size_t m, size_t n, size_t k,
                                         m_layout layout_a,
                                         m_layout layout_b)
    {
        // If empty, return default config
        if(all_configs.empty())
        {
            return DEFAULT_CONFIG_IDX;
        }

        // Logarithmic distance metric (better for matrix operations)
        auto size_distance = [](size_t m1, size_t n1, size_t k1,
                               size_t m2, size_t n2, size_t k2) -> double
        {
            double log_diff_m = std::log2(static_cast<double>(m1)) - std::log2(static_cast<double>(m2));
            double log_diff_n = std::log2(static_cast<double>(n1)) - std::log2(static_cast<double>(n2));
            double log_diff_k = std::log2(static_cast<double>(k1)) - std::log2(static_cast<double>(k2));
            return log_diff_m * log_diff_m + log_diff_n * log_diff_n + log_diff_k * log_diff_k;
        };

        // First try: find config with exact matching (A,B) layout and closest size
        double min_distance = std::numeric_limits<double>::max();
        size_t best_idx = DEFAULT_CONFIG_IDX;

        for(size_t i = 0; i < all_configs.size(); ++i)
        {
            const auto& entry = all_configs[i];

            // Check if (A,B) layout matches
            bool layout_match =
                (entry.layout_a == layout_a || entry.layout_a == m_layout::row_major) &&
                (entry.layout_b == layout_b || entry.layout_b == m_layout::row_major);

            if(layout_match)
            {
                double dist = size_distance(m, n, k, entry.m, entry.n, entry.k);
                if(dist < min_distance)
                {
                    min_distance = dist;
                    best_idx = i;
                }
            }
        }

        // If we found a match with right (A,B) layout, return it
        if(min_distance < std::numeric_limits<double>::max())
        {
            return all_configs[best_idx].config_idx;
        }

        // Second try: ignore layout and find closest size
        min_distance = std::numeric_limits<double>::max();

        for(size_t i = 0; i < all_configs.size(); ++i)
        {
            const auto& entry = all_configs[i];
            double dist = size_distance(m, n, k, entry.m, entry.n, entry.k);
            if(dist < min_distance)
            {
                min_distance = dist;
                best_idx = i;
            }
        }

        return all_configs[best_idx].config_idx;
    }

    // Find the best configuration for a given matrix size and (A,B) layout
    // Note: C layout is ignored as it doesn't affect the optimal kernel configuration
    constexpr size_t find_best_config(size_t m, size_t n, size_t k,
                                      m_layout layout_a,
                                      m_layout layout_b)
    {
        // First try exact match via switch-case (most efficient)
"""

    # Generate optimized switch-case for size lookup
    code += "        // First match by size\n"
    code += "        switch(m)\n"
    code += "        {\n"

    # Group by M dimension first
    m_groups = {}
    for size_key in sorted_sizes:
        M = size_key[0]
        if M not in m_groups:
            m_groups[M] = []
        m_groups[M].append(size_key)

    def generate_ab_layout_condition(ab_layout_key):
        """Generate (A,B) layout condition string from layout key."""
        a_layout, b_layout = ab_layout_key
        conditions = []
        if a_layout != "any":
            conditions.append(f"layout_a == m_layout::{a_layout}")
        if b_layout != "any":
            conditions.append(f"layout_b == m_layout::{b_layout}")
        return " && ".join(conditions) if conditions else ""

    def merge_ab_layout_conditions(ab_layouts_dict):
        """Merge (A,B) layout conditions that map to the same config."""
        # Group layouts by their config_idx
        config_to_layouts = defaultdict(list)
        for ab_layout_key, config_idx in ab_layouts_dict.items():
            config_to_layouts[config_idx].append(ab_layout_key)

        # Generate merged conditions
        merged_conditions = []
        for config_idx, ab_layout_keys in config_to_layouts.items():
            if len(ab_layout_keys) == 1:
                # Single condition
                condition = generate_ab_layout_condition(ab_layout_keys[0])
                if condition:
                    merged_conditions.append((f"({condition})", config_idx))
                else:
                    merged_conditions.append(("", config_idx))  # No condition needed
            else:
                # Multiple conditions that map to the same config - merge them
                individual_conditions = []
                for ab_layout_key in ab_layout_keys:
                    condition = generate_ab_layout_condition(ab_layout_key)
                    if condition:
                        individual_conditions.append(f"({condition})")

                if individual_conditions:
                    merged_condition = " || ".join(individual_conditions)
                    merged_conditions.append((f"({merged_condition})", config_idx))
                else:
                    # All layouts have no conditions - just return the config
                    merged_conditions.append(("", config_idx))

        return merged_conditions

    for M in sorted(m_groups.keys()):
        code += f"            case {M}:\n"
        code += "            {\n"

        # Group by N dimension
        n_groups = {}
        for size_key in m_groups[M]:
            N = size_key[1]
            if N not in n_groups:
                n_groups[N] = []
            n_groups[N].append(size_key)

        code += "                switch(n)\n"
        code += "                {\n"

        for N in sorted(n_groups.keys()):
            code += f"                    case {N}:\n"
            code += "                    {\n"

            # Finally, check K dimension
            k_sizes = sorted(n_groups[N], key=lambda x: x[2])

            code += "                        switch(k)\n"
            code += "                        {\n"

            for size_key in k_sizes:
                K = size_key[2]
                ab_layouts = size_ab_configs[size_key]

                code += f"                            case {K}:\n"
                code += "                            {\n"

                # Generate merged (A,B) layout checks for this specific size
                merged_conditions = merge_ab_layout_conditions(ab_layouts)

                # Sort by config_idx for consistent output
                merged_conditions.sort(key=lambda x: x[1])

                for condition_str, config_idx in merged_conditions:
                    if condition_str:
                        code += f"                                if({condition_str})\n"
                        code += "                                {\n"
                        code += f"                                    return {config_idx};\n"
                        code += "                                }\n"
                    else:
                        # No condition needed - this is the default/fallback case
                        code += f"                                return {config_idx};\n"

                code += "                                break;\n"
                code += "                            }\n"

            code += "                        }\n"
            code += "                        break;\n"
            code += "                    }\n"

        code += "                }\n"
        code += "                break;\n"
        code += "            }\n"

    code += """        }

        // If exact match not found, find closest configuration
        return find_closest_config(m, n, k, layout_a, layout_b);
    }

} // namespace detail

/**
 * @brief Get the optimal configuration parameters for a specific problem size and layout
 *
 * @param m Number of rows in matrices C and A
 * @param n Number of columns in matrices C and B
 * @param k Number of columns in matrix A / rows in matrix B
 * @param layout_c Layout of matrix C (ignored - same config used for both row/col major)
 * @param layout_a Layout of matrix A
 * @param layout_b Layout of matrix B
 * @return Tuned parameters for the given problem
 */
constexpr gemm_params get_gemm_params(size_t m, size_t n, size_t k,
                                       m_layout layout_c,
                                       m_layout layout_a,
                                       m_layout layout_b)
{
    // Find the best configuration for this problem (C layout ignored)
    const size_t config_idx = detail::find_best_config(m, n, k, layout_a, layout_b);

    // Get the configuration parameters
    const auto& config = detail::kernel_configs[config_idx];
    return gemm_params{
        std::get<0>(config),
        std::get<1>(config),
        std::get<2>(config),
        std::get<3>(config)
    };
}

/**
 * @brief Get the index of a kernel configuration in the configuration table
 *
 * @param params GEMM kernel parameters
 * @return Index of the kernel configuration
 */
constexpr size_t get_kernel_config_index(const gemm_params& params)
{
    for(size_t i = 0; i < detail::kernel_configs.size(); ++i)
    {
        const auto& config = detail::kernel_configs[i];
        if(std::get<0>(config) == params.warps_m &&
           std::get<1>(config) == params.warps_n &&
           std::get<2>(config) == params.warp_tile_m &&
           std::get<3>(config) == params.warp_tile_n)
        {
            return i;
        }
    }

    // Return the default config index if not found
    return detail::kernel_configs.size() - 1;
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP"""

    with open(output_file, 'w') as f:
        f.write(code)

def main():
    parser = argparse.ArgumentParser(description='Generate GEMM configuration header')
    parser.add_argument('config_file', type=str, help='Input JSON configuration file')
    parser.add_argument('output_file', type=str, help='Output header file')

    args = parser.parse_args()
    generate_config_header(args.config_file, args.output_file)

if __name__ == '__main__':
    main()
