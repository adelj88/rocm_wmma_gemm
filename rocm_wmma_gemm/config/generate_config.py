#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

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

    # Group configurations by matrix dimensions
    size_configs = {}
    for conf in config['configurations']:
        range_info = conf['range']
        M, N, K = range_info['M'], range_info['N'], range_info['K']

        # Create key for this matrix size
        size_key = (M, N, K)

        if size_key not in size_configs:
            size_configs[size_key] = {}

        # Create layout key from layout config
        layout_dict = conf['layout']
        layout_key = (
            layout_dict.get('A', 'any'),
            layout_dict.get('B', 'any'),
            layout_dict.get('C', 'any')
        )

        cfg = conf['config']
        config_tuple = (cfg['warps_m'], cfg['warps_n'],
                        cfg['warp_tile_m'], cfg['warp_tile_n'])
        config_idx = unique_configs.index(config_tuple)

        size_configs[size_key][layout_key] = config_idx

    # Sort sizes for search
    sorted_sizes = sorted(size_configs.keys())

    # Also create a flattened list of all configs for fallback search
    all_configs = []
    for size_key, layouts in size_configs.items():
        M, N, K = size_key
        for layout_key, config_idx in layouts.items():
            all_configs.append({
                'size': (M, N, K),
                'layout': layout_key,
                'config_idx': config_idx
            })

    # Generate code
    code = f"""// Auto-generated file - DO NOT EDIT
#ifndef ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP
#define ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP

#include <kernel/common.hpp>
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
    struct size_config_entry {
        size_t m, n, k;
        m_layout layout_a;
        m_layout layout_b;
        m_layout layout_c;
        size_t config_idx;
    };

    static constexpr std::array<size_config_entry, """ + str(len(all_configs)) + """> all_configs = {{
"""

    # Generate flattened config array for fallback search
    for i, entry in enumerate(all_configs):
        M, N, K = entry['size']
        a_layout, b_layout, c_layout = entry['layout']
        config_idx = entry['config_idx']

        # Convert layout strings to enum values
        a_enum = f"m_layout::{a_layout}" if a_layout != "any" else "m_layout::row_major"
        b_enum = f"m_layout::{b_layout}" if b_layout != "any" else "m_layout::row_major"
        c_enum = f"m_layout::{c_layout}" if c_layout != "any" else "m_layout::row_major"

        code += f"        {{{M}, {N}, {K}, {a_enum}, {b_enum}, {c_enum}, {config_idx}}}"
        code += "," if i < len(all_configs) - 1 else ""
        code += "\n"

    code += """    }};

    // Find closest configuration when exact match not found
    // This is a fallback mechanism only used when switch-case doesn't find a match
    constexpr size_t find_closest_config(size_t m, size_t n, size_t k,
                                       m_layout layout_c,
                                       m_layout layout_a,
                                       m_layout layout_b)
    {
        // If empty, return default config
        if (all_configs.empty()) {
            return DEFAULT_CONFIG_IDX;
        }

        // Logarithmic distance metric (better for matrix operations)
        auto size_distance = [](size_t m1, size_t n1, size_t k1,
                               size_t m2, size_t n2, size_t k2) -> double {
            double log_diff_m = std::log2(static_cast<double>(m1)) - std::log2(static_cast<double>(m2));
            double log_diff_n = std::log2(static_cast<double>(n1)) - std::log2(static_cast<double>(n2));
            double log_diff_k = std::log2(static_cast<double>(k1)) - std::log2(static_cast<double>(k2));
            return log_diff_m*log_diff_m + log_diff_n*log_diff_n + log_diff_k*log_diff_k;
        };

        // First try: find config with exact matching layout and closest size
        double min_distance = std::numeric_limits<double>::max();
        size_t best_idx = DEFAULT_CONFIG_IDX;

        for (size_t i = 0; i < all_configs.size(); ++i) {
            const auto& entry = all_configs[i];

            // Check if layout matches
            bool layout_match =
                (entry.layout_a == layout_a || entry.layout_a == m_layout::row_major) &&
                (entry.layout_b == layout_b || entry.layout_b == m_layout::row_major) &&
                (entry.layout_c == layout_c || entry.layout_c == m_layout::row_major);

            if (layout_match) {
                double dist = size_distance(m, n, k, entry.m, entry.n, entry.k);
                if (dist < min_distance) {
                    min_distance = dist;
                    best_idx = i;
                }
            }
        }

        // If we found a match with right layout, return it
        if (min_distance < std::numeric_limits<double>::max()) {
            return all_configs[best_idx].config_idx;
        }

        // Second try: ignore layout and find closest size
        min_distance = std::numeric_limits<double>::max();

        for (size_t i = 0; i < all_configs.size(); ++i) {
            const auto& entry = all_configs[i];
            double dist = size_distance(m, n, k, entry.m, entry.n, entry.k);
            if (dist < min_distance) {
                min_distance = dist;
                best_idx = i;
            }
        }

        return all_configs[best_idx].config_idx;
    }

    // Find the best configuration for a given matrix size and layout
    constexpr size_t find_best_config(size_t m, size_t n, size_t k,
                                      m_layout layout_c,
                                      m_layout layout_a,
                                      m_layout layout_b)
    {
        // First try exact match via switch-case (most efficient)
"""

    # Generate switch-case for size lookup
    code += "        // First match by size\n"
    code += "        switch (m)\n"
    code += "        {\n"

    # Group by M dimension first
    m_groups = {}
    for size_key in sorted_sizes:
        M = size_key[0]
        if M not in m_groups:
            m_groups[M] = []
        m_groups[M].append(size_key)

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

        code += "                switch (n)\n"
        code += "                {\n"

        for N in sorted(n_groups.keys()):
            code += f"                    case {N}:\n"
            code += "                    {\n"

            # Finally, check K dimension
            k_sizes = sorted(n_groups[N], key=lambda x: x[2])

            code += "                        switch (k)\n"
            code += "                        {\n"

            for size_key in k_sizes:
                K = size_key[2]
                layouts = size_configs[size_key]

                code += f"                            case {K}:\n"
                code += "                            {\n"

                # Generate layout checks for this specific size
                for layout_key, config_idx in layouts.items():
                    a_layout, b_layout, c_layout = layout_key

                    # Build layout condition
                    conditions = []
                    if a_layout != "any":
                        conditions.append(f"layout_a == m_layout::{a_layout}")
                    if b_layout != "any":
                        conditions.append(f"layout_b == m_layout::{b_layout}")
                    if c_layout != "any":
                        conditions.append(f"layout_c == m_layout::{c_layout}")

                    # If we have conditions, write if statement
                    if conditions:
                        condition = " && ".join(conditions)
                        code += f"                                if ({condition})\n"
                        code += "                                {\n"
                        code += f"                                    return {config_idx};\n"
                        code += "                                }\n"
                    else:
                        # If no conditions, always use this config
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
        return find_closest_config(m, n, k, layout_c, layout_a, layout_b);
    }

} // namespace detail

/**
 * @brief Get the optimal configuration parameters for a specific problem size and layout
 *
 * @param m Number of rows in matrices C and A
 * @param n Number of columns in matrices C and B
 * @param k Number of columns in matrix A / rows in matrix B
 * @param layout_c Layout of matrix C
 * @param layout_a Layout of matrix A
 * @param layout_b Layout of matrix B
 * @return Tuned parameters for the given problem
 */
constexpr gemm_params get_gemm_params(size_t m, size_t n, size_t k,
                                     m_layout layout_c,
                                     m_layout layout_a,
                                     m_layout layout_b)
{
    // Find the best configuration for this problem
    const size_t config_idx = detail::find_best_config(m, n, k, layout_c, layout_a, layout_b);

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
    for (size_t i = 0; i < detail::kernel_configs.size(); ++i)
    {
        const auto& config = detail::kernel_configs[i];
        if (std::get<0>(config) == params.warps_m &&
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
