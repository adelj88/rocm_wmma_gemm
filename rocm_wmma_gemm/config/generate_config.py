#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

def generate_layout_condition(layout, exclude=None):
    # For M,N <= 1024, we need special handling to match the original logic
    if 'exclude' in layout:
        exclude = layout['exclude']
        if "combination" in exclude:
            # For the 1024 < M,N <= 2048 case
            subconditions = []
            for combo in exclude["combination"]:
                combo_conditions = [f"layout_{m} == m_layout::{l}"
                                 for m, l in combo.items()]
                subconditions.append(f"!({' && '.join(combo_conditions)})")
            return f"({' && '.join(subconditions)})"
        else:
            # For the M,N <= 1024 case
            blocked_layout = []
            for matrix, layout_type in exclude.items():
                blocked_layout.append(f"layout_{matrix} == m_layout::{layout_type}")
            return f"!({' && '.join(blocked_layout)})"

    # For specific layout requirements
    conditions = []
    for matrix in ['A', 'B', 'C']:
        if layout.get(matrix, "any") != "any":
            conditions.append(f"layout_{matrix} == m_layout::{layout[matrix]}")

    if not conditions:
        return "true"
    return ' && '.join(conditions)

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
        range_m = conf['range']['M']
        range_n = conf['range']['N']
        layout = conf['layout']

        # Special handling for M,N <= 1024 case to match original logic
        range_conditions = []
        if range_m['max'] == 1024 and range_n['max'] == 1024:
            range_conditions = [f"M <= {range_m['max']}", f"N <= {range_n['max']}"]
        else:
            if range_m['min'] > 1:
                range_conditions.append(f"M >= {range_m['min']}")
            if range_m['max'] != -1:
                range_conditions.append(f"M <= {range_m['max']}")
            if range_n['min'] > 1:
                range_conditions.append(f"N >= {range_n['min']}")
            if range_n['max'] != -1:
                range_conditions.append(f"N <= {range_n['max']}")

        # Generate layout condition
        layout_condition = generate_layout_condition(layout)

        # Combine conditions to match original style
        if layout_condition != "true":
            if range_conditions:
                condition_str = f"({layout_condition}) && {' && '.join(range_conditions)}"
            else:
                condition_str = layout_condition
        else:
            condition_str = ' && '.join(range_conditions) if range_conditions else "true"

        if condition_str != "true":
            cfg = conf['config']
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
