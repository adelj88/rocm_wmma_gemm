#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

def generate_layout_condition(layout):
    """Generate layout condition code"""
    conditions = []

    # Handle basic layout requirements
    for matrix in ['A', 'B', 'C']:
        if layout.get(matrix, "any") != "any":
            conditions.append(f"layout_{matrix} == m_layout::{layout[matrix]}")

    if not conditions:
        return "true"
    return " && ".join(conditions)

def generate_config_header(config_file, output_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract unique configurations first
    unique_configs = set()
    for conf in config['configurations']:
        cfg = conf['config']
        unique_configs.add((cfg['warps_m'], cfg['warps_n'],
                          cfg['warp_tile_m'], cfg['warp_tile_n']))

    # Always add default config if not present
    default_config = (4, 4, 4, 4)
    if default_config not in unique_configs:
        unique_configs.add(default_config)

    # Sort configs for stable output
    unique_configs = sorted(list(unique_configs))
    num_configs = len(unique_configs)

    code = f"""// Auto-generated file - DO NOT EDIT
#ifndef ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP
#define ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP

#include <kernel/common.hpp>
#include <array>
#include <tuple>

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

// Unique kernel configurations from the JSON
namespace detail
{{
    using kernel_config = std::tuple<int, int, int, int>; // warps_m, warps_n, warp_tile_m, warp_tile_n

    template<typename T>
    constexpr size_t tuple_index(const std::array<T, KERNEL_VARIANTS>& arr, const T& value)
    {{
        for (size_t i = 0; i < arr.size(); ++i)
        {{
            if (arr[i] == value) return i;
        }}
        return arr.size() - 1; // Return last (default) config if not found
    }}

    // All unique kernel configurations
    static constexpr std::array<kernel_config, KERNEL_VARIANTS> kernel_configs = {{
"""

    # Generate config array
    for i, (wm, wn, wtm, wtn) in enumerate(unique_configs):
        code += f"        std::tuple<int, int, int, int>{{{wm}, {wn}, {wtm}, {wtn}}}"
        code += "," if i < len(unique_configs) - 1 else ""
        code += f" // Config {i}\n"

    code += """    };
} // namespace detail

inline gemm_params get_gemm_params(size_t M, size_t N, size_t K,
                                m_layout layout_C,
                                m_layout layout_A,
                                m_layout layout_B)
{
    gemm_params params{4, 4, 4, 4}; // Default configuration
"""

    # Generate conditions for each configuration
    for idx, conf in enumerate(config['configurations']):
        range_info = conf['range']
        layout = conf['layout']
        cfg = conf['config']

        # Generate size condition
        size_conditions = []
        for dim, val in range_info.items():
            size_conditions.append(f"{dim} == {val}")

        # Generate layout condition
        layout_conditions = []
        for matrix in ['A', 'B', 'C']:
            if layout.get(matrix, "any") != "any":
                layout_conditions.append(f"layout_{matrix} == m_layout::{layout[matrix]}")

        # Combine conditions
        conditions = size_conditions + layout_conditions
        if conditions:
            condition_str = ' && '.join(conditions)
            config_tuple = (cfg['warps_m'], cfg['warps_n'],
                          cfg['warp_tile_m'], cfg['warp_tile_n'])
            config_idx = unique_configs.index(config_tuple)

            code += f"""
    if ({condition_str})
    {{
        const auto& config = detail::kernel_configs[{config_idx}];
        return gemm_params{{std::get<0>(config), std::get<1>(config),
                         std::get<2>(config), std::get<3>(config)}};
    }}"""

    code += """
    // Return default configuration
    const auto& config = detail::kernel_configs[detail::kernel_configs.size() - 1];
    return gemm_params{std::get<0>(config), std::get<1>(config),
                     std::get<2>(config), std::get<3>(config)};
}

// Get the index of a kernel configuration
inline constexpr size_t get_kernel_config_index(const gemm_params& params)
{
    return detail::tuple_index(detail::kernel_configs,
        std::tuple<int, int, int, int>{params.warps_m, params.warps_n,
                                      params.warp_tile_m, params.warp_tile_n});
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
