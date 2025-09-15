#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import defaultdict

def generate_config_header(config_file, output_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract unique configurations (now 6 parameters including use_direct_write)
    unique_configs = set()
    for conf in config['configurations']:
        cfg = conf['config']
        unique_configs.add((cfg['warps_m'], cfg['warps_n'],
                            cfg['warp_tile_m'], cfg['warp_tile_n'],
                            cfg['bits'], cfg['use_direct_write']))

    # Always add default configs if not present
    default_configs = [
        (4, 4, 4, 4, 256, False),
        (4, 4, 4, 4, 256, True)
    ]
    for default_config in default_configs:
        if default_config not in unique_configs:
            unique_configs.add(default_config)

    # Sort configs for stable output and indexing
    unique_configs = sorted(list(unique_configs))
    num_configs = len(unique_configs)

    # Group configurations by matrix dimensions and ALL THREE layouts (A, B, C)
    size_layout_configs = {}
    for conf in config['configurations']:
        range_info = conf['range']
        M, N, K = range_info['M'], range_info['N'], range_info['K']

        # Create key for this matrix size
        size_key = (M, N, K)

        # Create layout key from (A, B, C) - all three matter now
        layout_dict = conf['layout']
        layout_key = (
            layout_dict.get('A', 'any'),
            layout_dict.get('B', 'any'),
            layout_dict.get('C', 'any')  # Include C layout
        )

        if size_key not in size_layout_configs:
            size_layout_configs[size_key] = {}

        cfg = conf['config']
        config_tuple = (cfg['warps_m'], cfg['warps_n'],
                        cfg['warp_tile_m'], cfg['warp_tile_n'],
                        cfg['bits'], cfg['use_direct_write'])
        config_idx = unique_configs.index(config_tuple)

        size_layout_configs[size_key][layout_key] = config_idx

    # Create sorted configuration list for binary search
    sorted_configs = []
    for size_key, layouts in size_layout_configs.items():
        M, N, K = size_key
        for layout_key, config_idx in layouts.items():
            a_layout, b_layout, c_layout = layout_key
            sorted_configs.append((M, N, K, a_layout, b_layout, c_layout, config_idx))

    # Sort by (M, N, K, A, B, C) for binary search
    def sort_key(x):
        M, N, K, a_layout, b_layout, c_layout, config_idx = x
        # Convert layouts to comparable values: row_major=0, col_major=1
        a_val = 0 if a_layout == "row_major" else 1
        b_val = 0 if b_layout == "row_major" else 1
        c_val = 0 if c_layout == "row_major" else 1
        return (M, N, K, a_val, b_val, c_val)

    sorted_configs.sort(key=sort_key)

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
#include <algorithm>

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
    int bits;
    bool use_direct_write;
}};

namespace detail
{{
    // Kernel configuration tuple (now 6 parameters)
    using kernel_config = std::tuple<int, int, int, int, int, bool>;

    // All unique kernel configurations
    static constexpr std::array<kernel_config, KERNEL_VARIANTS> kernel_configs = {{
"""

    # Generate config array
    for i, (wm, wn, wtm, wtn, bits, direct) in enumerate(unique_configs):
        code += f"        std::tuple<int, int, int, int, int, bool>{{{wm}, {wn}, {wtm}, {wtn}, {bits}, {str(direct).lower()}}}"
        code += "," if i < len(unique_configs) - 1 else ""
        code += f" // Config {i}: warps({wm},{wn}) tiles({wtm},{wtn}) bits({bits}) direct({direct})\n"

    code += f"""    }};

    // Default config (last in the array)
    static constexpr size_t DEFAULT_CONFIG_IDX = KERNEL_VARIANTS - 1;

    // Configuration lookup key (now includes layout_c)
    struct config_key
    {{
        size_t m, n, k;
        m_layout layout_a, layout_b, layout_c;

        constexpr bool operator<(const config_key& other) const
        {{
            if(m != other.m) return m < other.m;
            if(n != other.n) return n < other.n;
            if(k != other.k) return k < other.k;
            if(layout_a != other.layout_a) return layout_a < other.layout_a;
            if(layout_b != other.layout_b) return layout_b < other.layout_b;
            return layout_c < other.layout_c;
        }}

        constexpr bool operator==(const config_key& other) const
        {{
            return m == other.m && n == other.n && k == other.k &&
                   layout_a == other.layout_a && layout_b == other.layout_b &&
                   layout_c == other.layout_c;
        }}
    }};

    // Sorted configuration map for binary search
    static constexpr std::array<std::pair<config_key, size_t>, {len(sorted_configs)}> sorted_config_map = {{{{
"""

    # Generate sorted config array
    for i, (M, N, K, a_layout, b_layout, c_layout, config_idx) in enumerate(sorted_configs):
        # Convert layout strings to enum values
        a_enum = f"m_layout::{a_layout}" if a_layout != "any" else "m_layout::row_major"
        b_enum = f"m_layout::{b_layout}" if b_layout != "any" else "m_layout::row_major"
        c_enum = f"m_layout::{c_layout}" if c_layout != "any" else "m_layout::row_major"

        code += f"        {{{{{M}, {N}, {K}, {a_enum}, {b_enum}, {c_enum}}}, {config_idx}}}"
        code += "," if i < len(sorted_configs) - 1 else ""
        code += f" // {M}x{N}x{K}, A={a_layout}, B={b_layout}, C={c_layout}\n"

    code += """    }};

    // Find closest configuration when exact match not found
    constexpr size_t find_closest_config(size_t m, size_t n, size_t k,
                                         m_layout layout_a,
                                         m_layout layout_b,
                                         m_layout layout_c)
    {
        // If empty, return default config
        if(sorted_config_map.empty())
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

        // Find config with exact matching (A,B,C) layout and closest size
        double min_distance = std::numeric_limits<double>::max();
        size_t best_idx = DEFAULT_CONFIG_IDX;

        for(size_t i = 0; i < sorted_config_map.size(); ++i)
        {
            const auto& entry = sorted_config_map[i];
            const auto& key = entry.first;

            // Check if (A,B,C) layout matches exactly
            if(key.layout_a == layout_a && key.layout_b == layout_b && key.layout_c == layout_c)
            {
                double dist = size_distance(m, n, k, key.m, key.n, key.k);
                if(dist < min_distance)
                {
                    min_distance = dist;
                    best_idx = i;
                }
            }
        }

        // If we found a match with right (A,B,C) layout, return it
        if(min_distance < std::numeric_limits<double>::max())
        {
            return sorted_config_map[best_idx].second;
        }

        // Fallback: find closest size regardless of layout
        min_distance = std::numeric_limits<double>::max();

        for(size_t i = 0; i < sorted_config_map.size(); ++i)
        {
            const auto& entry = sorted_config_map[i];
            const auto& key = entry.first;
            double dist = size_distance(m, n, k, key.m, key.n, key.k);
            if(dist < min_distance)
            {
                min_distance = dist;
                best_idx = i;
            }
        }

        return sorted_config_map[best_idx].second;
    }

    // Find the best configuration using binary search
    constexpr size_t find_best_config(size_t m, size_t n, size_t k,
                                      m_layout layout_a,
                                      m_layout layout_b,
                                      m_layout layout_c)
    {
        config_key target{m, n, k, layout_a, layout_b, layout_c};

        // Binary search using std::lower_bound
        auto it = std::lower_bound(
            sorted_config_map.begin(),
            sorted_config_map.end(),
            std::make_pair(target, size_t(0)),
            [](const auto& a, const auto& b)
            {
                return a.first < b.first;
            }
        );

        // Check if we found an exact match
        if(it != sorted_config_map.end() && it->first == target)
        {
            return it->second;
        }

        // Fall back to closest match
        return find_closest_config(m, n, k, layout_a, layout_b, layout_c);
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
    // Find the best configuration for this problem (now includes C layout)
    const size_t config_idx = detail::find_best_config(m, n, k, layout_a, layout_b, layout_c);

    // Get the configuration parameters
    const auto& config = detail::kernel_configs[config_idx];
    return gemm_params{
        std::get<0>(config),  // warps_m
        std::get<1>(config),  // warps_n
        std::get<2>(config),  // warp_tile_m
        std::get<3>(config),  // warp_tile_n
        std::get<4>(config),  // bits
        std::get<5>(config)   // use_direct_write
    };
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP"""

    with open(output_file, 'w') as f:
        f.write(code)


def generate_kernel_sources(config_file, output_dir):
    """Generate separate source files for each (config, layout) combination from JSON"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Type pairs to instantiate
    type_pairs = [
        ('half', 'half'),
        ('float', 'half'),
        ('__hip_bfloat16', '__hip_bfloat16'),
        ('float', '__hip_bfloat16')
    ]

    # Track unique (config, layout) combinations
    seen_combinations = set()
    file_list = []
    file_index = 0

    for conf in config['configurations']:
        cfg = conf['config']
        layout = conf['layout']
        range_info = conf['range']

        # Create key for this combination
        config_key = (
            cfg['warps_m'], cfg['warps_n'],
            cfg['warp_tile_m'], cfg['warp_tile_n'],
            cfg['bits'], cfg['use_direct_write'],
            layout['A'], layout['B'], layout['C']
        )

        # Skip if we've already generated this combination
        if config_key in seen_combinations:
            continue
        seen_combinations.add(config_key)

        wm, wn, wtm, wtn, bits, direct = config_key[:6]
        layout_a, layout_b, layout_c = config_key[6:]

        filename = f"kernel_inst_{file_index}.cpp"
        filepath = output_path / filename
        file_list.append(filename)

        # Generate the source file with unique kernel name
        code = f"""// Auto-generated kernel instantiation file - DO NOT EDIT
// Config: warps({wm},{wn}) tiles({wtm},{wtn}) bits({bits}) direct({direct})
// Layout: A={layout_a}, B={layout_b}, C={layout_c}
// Size hint: {range_info['M']}x{range_info['N']}x{range_info['K']}

// Rename kernel to avoid ODR violations across compilation units
#define kernel_gemm kernel_gemm_inst_{file_index}
#include <rocm_wmma_gemm/kernel/kernel.hpp>
#undef kernel_gemm

namespace rocm_wmma_gemm
{{

// Extern C getters for each type pair
"""

        for t_type, u_type in type_pairs:
            # Create a mangled name for the getter
            t_name = t_type.replace('__hip_bfloat16', 'bf16').replace('half', 'f16').replace('float', 'f32')
            u_name = u_type.replace('__hip_bfloat16', 'bf16').replace('half', 'f16').replace('float', 'f32')
            layout_str = f"{layout_a[0]}{layout_b[0]}{layout_c[0]}"  # e.g., "rrr" for A=row_major, B=row_major, C=row_major

            code += f"""extern "C" void* get_kernel_inst_{file_index}_{t_name}_{u_name}_{layout_str}() {{
    return (void*)&kernel_gemm_inst_{file_index}<{t_type}, {u_type},
        m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
        {wm}, {wn}, {wtm}, {wtn}, {bits}, {str(direct).lower()}>;
}}

"""

        code += """} // namespace rocm_wmma_gemm
"""

        with open(filepath, 'w') as f:
            f.write(code)

        file_index += 1

    # Generate a file list for CMake
    filelist_path = output_path / "kernel_sources.txt"
    with open(filelist_path, 'w') as f:
        for filename in file_list:
            f.write(f"{filename}\n")

    # Generate a CMake file that lists all kernel sources
    cmake_file = output_path / "kernel_sources.cmake"
    with open(cmake_file, 'w') as f:
        f.write("# Auto-generated list of kernel source files\n")
        f.write("set(KERNEL_INST_SOURCES\n")
        for filename in file_list:
            f.write(f"    ${{CMAKE_CURRENT_BINARY_DIR}}/src/kernel_inst/{filename}\n")
        f.write(")\n")

    print(f"Generated {len(file_list)} kernel source files in {output_dir}")
    print(f"Each file instantiates 1 (config, layout) × 4 type pairs = 4 kernels")
    print(f"Total kernels: {len(file_list)} × 4 = {len(file_list) * 4}")

    # Generate kernel lookup file
    generate_kernel_lookup(config_file, output_path, file_list)

    return file_list


def generate_kernel_lookup(config_file, output_dir, file_list):
    """Generate the kernel lookup implementation with static table"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract unique configurations
    unique_configs = set()
    for conf in config['configurations']:
        cfg = conf['config']
        unique_configs.add((cfg['warps_m'], cfg['warps_n'],
                            cfg['warp_tile_m'], cfg['warp_tile_n'],
                            cfg['bits'], cfg['use_direct_write']))

    # Add default configs
    default_configs = [
        (4, 4, 4, 4, 256, False),
        (4, 4, 4, 4, 256, True)
    ]
    for default_config in default_configs:
        if default_config not in unique_configs:
            unique_configs.add(default_config)

    unique_configs = sorted(list(unique_configs))
    num_configs = len(unique_configs)

    # Type pairs
    type_pairs = [
        ('half', 'half', 'f16', 'f16'),
        ('float', 'half', 'f32', 'f16'),
        ('__hip_bfloat16', '__hip_bfloat16', 'bf16', 'bf16'),
        ('float', '__hip_bfloat16', 'f32', 'bf16')
    ]

    # Layout combinations
    layouts = [
        ('row_major', 'row_major', 'row_major', 'rrr'),
        ('row_major', 'row_major', 'col_major', 'rrc'),
        ('row_major', 'col_major', 'row_major', 'rcr'),
        ('row_major', 'col_major', 'col_major', 'rcc'),
        ('col_major', 'row_major', 'row_major', 'crr'),
        ('col_major', 'row_major', 'col_major', 'crc'),
        ('col_major', 'col_major', 'row_major', 'ccr'),
        ('col_major', 'col_major', 'col_major', 'ccc'),
    ]

    # Build mapping from (config_tuple, layout_tuple) -> file_index
    config_layout_to_file = {}
    seen_combinations = set()
    file_index = 0

    for conf in config['configurations']:
        cfg = conf['config']
        layout = conf['layout']

        config_key = (
            cfg['warps_m'], cfg['warps_n'],
            cfg['warp_tile_m'], cfg['warp_tile_n'],
            cfg['bits'], cfg['use_direct_write'],
            layout['A'], layout['B'], layout['C']
        )

        if config_key in seen_combinations:
            continue
        seen_combinations.add(config_key)

        config_tuple = config_key[:6]
        layout_tuple = (layout['A'], layout['B'], layout['C'])
        config_layout_to_file[(config_tuple, layout_tuple)] = file_index

        file_index += 1

    filepath = output_dir / "kernel_lookup.cpp"

    code = """// Auto-generated kernel lookup - DO NOT EDIT
#include <rocm_wmma_gemm/kernel/common.hpp>
#include <cstddef>

namespace rocm_wmma_gemm
{

// Forward declare extern C getters (only those that actually exist)
"""

    # Declare only the getters that actually exist
    # Build reverse mapping: file_idx -> (config_tuple, layout_tuple)
    file_to_config_layout = {}
    for (config_tuple, layout_tuple), file_idx in config_layout_to_file.items():
        file_to_config_layout[file_idx] = (config_tuple, layout_tuple)

    # Declare getters only for the layouts that exist in each file
    for file_idx in range(file_index):
        if file_idx in file_to_config_layout:
            _, layout_tuple = file_to_config_layout[file_idx]
            layout_a, layout_b, layout_c = layout_tuple
            # Find the layout_str for this layout_tuple
            layout_str = None
            for la, lb, lc, ls in layouts:
                if (la, lb, lc) == layout_tuple:
                    layout_str = ls
                    break

            if layout_str:
                # Declare getters for all type pairs for this one layout
                for _, _, t_name, u_name in type_pairs:
                    code += f'extern "C" void* get_kernel_inst_{file_idx}_{t_name}_{u_name}_{layout_str}();\n'

    code += f"""
// Static kernel lookup table: [config_idx][type_idx][layout_idx]
// config_idx: 0 to {num_configs-1}
// type_idx: 0=half-half, 1=float-half, 2=bf16-bf16, 3=float-bf16
// layout_idx: 0=rrr, 1=rrc, 2=rcr, 3=rcc, 4=crr, 5=crc, 6=ccr, 7=ccc
static void* kernel_table[{num_configs}][4][8] = {{
"""

    # Generate the table initialization
    for config_idx, config_tuple in enumerate(unique_configs):
        code += f"    // Config {config_idx}: warps({config_tuple[0]},{config_tuple[1]}) tiles({config_tuple[2]},{config_tuple[3]}) bits({config_tuple[4]}) direct({config_tuple[5]})\n"
        code += "    {\n"

        for type_idx, (_, _, t_name, u_name) in enumerate(type_pairs):
            code += "        {"

            for layout_idx, (layout_a, layout_b, layout_c, layout_str) in enumerate(layouts):
                layout_tuple = (layout_a, layout_b, layout_c)
                key = (config_tuple, layout_tuple)

                if key in config_layout_to_file:
                    file_idx = config_layout_to_file[key]
                    # Only reference the getter if this file actually provides this layout
                    _, file_layout = file_to_config_layout[file_idx]
                    if file_layout == layout_tuple:
                        code += f"get_kernel_inst_{file_idx}_{t_name}_{u_name}_{layout_str}()"
                    else:
                        code += "nullptr"
                else:
                    code += "nullptr"

                if layout_idx < 7:
                    code += ", "

            code += "}"
            if type_idx < 3:
                code += ","
            code += "\n"

        code += "    }"
        if config_idx < num_configs - 1:
            code += ","
        code += "\n"

    code += """};

// Simple lookup function
void* lookup_kernel(size_t config_idx, size_t type_idx, size_t layout_idx)
{
    return kernel_table[config_idx][type_idx][layout_idx];
}

} // namespace rocm_wmma_gemm
"""

    with open(filepath, 'w') as f:
        f.write(code)

    print(f"Generated kernel lookup table in {filepath}")


def generate_dispatch_table(config_file, kernel_dir, include_dir):
    """Generate the compile-time dispatch table implementation"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract unique configurations
    unique_configs = set()
    for conf in config['configurations']:
        cfg = conf['config']
        unique_configs.add((cfg['warps_m'], cfg['warps_n'],
                            cfg['warp_tile_m'], cfg['warp_tile_n'],
                            cfg['bits'], cfg['use_direct_write']))

    # Add default configs
    default_configs = [
        (4, 4, 4, 4, 256, False),
        (4, 4, 4, 4, 256, True)
    ]
    for default_config in default_configs:
        if default_config not in unique_configs:
            unique_configs.add(default_config)

    # Sort configs for stable indexing
    unique_configs = sorted(list(unique_configs))

    # Generate dispatch implementation in the include directory
    filepath = Path(include_dir) / "kernel_dispatch_impl.hpp"

    code = """// Auto-generated dispatch table - DO NOT EDIT
#ifndef ROCM_WMMA_GEMM_KERNEL_DISPATCH_IMPL_HPP
#define ROCM_WMMA_GEMM_KERNEL_DISPATCH_IMPL_HPP

#include <rocm_wmma_gemm/kernel_dispatch.hpp>

namespace rocm_wmma_gemm
{

// Dispatch implementation for each type/layout combination
"""

    # Generate dispatch for each type pair and layout combination
    type_pairs = [
        ('half', 'half'),
        ('float', 'half'),
        ('__hip_bfloat16', '__hip_bfloat16'),
        ('float', '__hip_bfloat16')
    ]

    layouts = [
        ('row_major', 'row_major', 'row_major'),
        ('row_major', 'row_major', 'col_major'),
        ('row_major', 'col_major', 'row_major'),
        ('row_major', 'col_major', 'col_major'),
        ('col_major', 'row_major', 'row_major'),
        ('col_major', 'row_major', 'col_major'),
        ('col_major', 'col_major', 'row_major'),
        ('col_major', 'col_major', 'col_major'),
    ]

    for t_type, u_type in type_pairs:
        for layout_c, layout_a, layout_b in layouts:
            code += f"""
template<>
typename kernel_dispatch<{t_type}, {u_type}, m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b}>::kernel_func_ptr
kernel_dispatch<{t_type}, {u_type}, m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b}>::dispatch_by_params(
    int warps_m, int warps_n, int warp_tile_m, int warp_tile_n, int bits, bool use_direct_write)
{{
"""

            # Generate if-else chain for each config
            for i, (wm, wn, wtm, wtn, b, direct) in enumerate(unique_configs):
                if_keyword = "if" if i == 0 else "else if"
                code += f"""    {if_keyword}(warps_m == {wm} && warps_n == {wn} && warp_tile_m == {wtm} &&
       warp_tile_n == {wtn} && bits == {b} && use_direct_write == {str(direct).lower()})
    {{
        return &kernel_gemm<{t_type}, {u_type}, m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
                           {wm}, {wn}, {wtm}, {wtn}, {b}, {str(direct).lower()}>;
    }}
"""

            code += """    return nullptr;
}

"""

    code += """} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_KERNEL_DISPATCH_IMPL_HPP
"""

    with open(filepath, 'w') as f:
        f.write(code)

    print(f"Generated dispatch table in {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate WMMA GEMM configuration header and kernel sources')
    parser.add_argument('config_file', type=str, help='Input JSON configuration file')
    parser.add_argument('output_file', type=str, help='Output header file')
    parser.add_argument('--kernel-dir', type=str, help='Output directory for kernel source files')

    args = parser.parse_args()

    # Always generate the config header
    generate_config_header(args.config_file, args.output_file)

    # Generate kernel sources if directory specified
    if args.kernel_dir:
        # Generate kernel sources
        generate_kernel_sources(args.config_file, args.kernel_dir)


if __name__ == '__main__':
    main()
