#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import defaultdict

def load_configs(f16_config_file, f32_config_file=None):
    """Load one or two config files and tag each entry with its type_group.
    type_group 0 = f16/bf16 (from f16 file), type_group 1 = f32 (from f32 file).
    Returns (all_confs, unique_configs_sorted).
    """
    all_confs = []

    with open(f16_config_file, 'r') as f:
        cfg = json.load(f)
    for conf in cfg['configurations']:
        all_confs.append((conf, 0))  # type_group 0

    if f32_config_file:
        with open(f32_config_file, 'r') as f:
            cfg = json.load(f)
        for conf in cfg['configurations']:
            all_confs.append((conf, 1))  # type_group 1

    unique_configs = set()
    for conf, _ in all_confs:
        c = conf['config']
        unique_configs.add((c['warps_m'], c['warps_n'],
                            c['warp_tile_m'], c['warp_tile_n'],
                            c['swizzle'], c['bits']))

    default_configs = [(4, 4, 4, 4, 8, 256)]
    for dc in default_configs:
        if dc not in unique_configs:
            unique_configs.add(dc)

    return all_confs, sorted(list(unique_configs))


def generate_config_header(f16_config_file, output_file, f32_config_file=None):
    all_confs, unique_configs = load_configs(f16_config_file, f32_config_file)
    num_configs = len(unique_configs)

    # Build sorted_config_map entries: one per (type_group, M, N, K, layouts).
    # type_group is included in config_key so f32 entries are distinct from f16.
    size_layout_configs = {}
    for conf, type_group in all_confs:
        range_info = conf['range']
        M, N, K = range_info['M'], range_info['N'], range_info['K']
        layout_dict = conf['layout']
        layout_key = (
            layout_dict.get('A', 'any'),
            layout_dict.get('B', 'any'),
            layout_dict.get('C', 'any'),
        )
        c = conf['config']
        config_tuple = (c['warps_m'], c['warps_n'],
                        c['warp_tile_m'], c['warp_tile_n'],
                        c['swizzle'], c['bits'])
        config_idx = unique_configs.index(config_tuple)
        size_layout_configs.setdefault((M, N, K), {})[(type_group, layout_key)] = config_idx

    sorted_configs = []
    for (M, N, K), tl_map in size_layout_configs.items():
        for (type_group, (a_layout, b_layout, c_layout)), config_idx in tl_map.items():
            sorted_configs.append((M, N, K, type_group, a_layout, b_layout, c_layout, config_idx))

    def sort_key(x):
        M, N, K, tg, a, b, c, _ = x
        return (M, N, K, tg,
                0 if a == 'row_major' else 1,
                0 if b == 'row_major' else 1,
                0 if c == 'row_major' else 1)

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
    int swizzle;
    int bits;
}};

namespace detail
{{
    // Kernel configuration tuple
    using kernel_config = std::tuple<int, int, int, int, int, int>;

    // All unique kernel configurations
    static constexpr std::array<kernel_config, KERNEL_VARIANTS> kernel_configs = {{
"""

    # Generate config array
    for i, (wm, wn, wtm, wtn, swizzle, bits) in enumerate(unique_configs):
        code += f"        std::tuple<int, int, int, int, int, int>{{{wm}, {wn}, {wtm}, {wtn}, {swizzle}, {bits}}}"
        code += "," if i < len(unique_configs) - 1 else ""
        code += f" // Config {i}: warps({wm},{wn}) tiles({wtm},{wtn}) swizzle({swizzle}) bits({bits})\n"

    code += f"""    }};

    // Default config (last in the array)
    static constexpr size_t DEFAULT_CONFIG_IDX = KERNEL_VARIANTS - 1;

    // Configuration lookup key.
    // type_group: 0 = f16/bf16, 1 = f32 accumulator.
    struct config_key
    {{
        size_t m, n, k;
        size_t type_group;
        m_layout layout_a, layout_b, layout_c;

        constexpr bool operator<(const config_key& other) const
        {{
            if(m != other.m) return m < other.m;
            if(n != other.n) return n < other.n;
            if(k != other.k) return k < other.k;
            if(type_group != other.type_group) return type_group < other.type_group;
            if(layout_a != other.layout_a) return layout_a < other.layout_a;
            if(layout_b != other.layout_b) return layout_b < other.layout_b;
            return layout_c < other.layout_c;
        }}

        constexpr bool operator==(const config_key& other) const
        {{
            return m == other.m && n == other.n && k == other.k &&
                   type_group == other.type_group &&
                   layout_a == other.layout_a && layout_b == other.layout_b &&
                   layout_c == other.layout_c;
        }}
    }};

    // Sorted configuration map for binary search
    static constexpr std::array<std::pair<config_key, size_t>, {len(sorted_configs)}> sorted_config_map = {{{{
"""

    # Generate sorted config array
    for i, (M, N, K, type_group, a_layout, b_layout, c_layout, config_idx) in enumerate(sorted_configs):
        a_enum = f"m_layout::{a_layout}" if a_layout != "any" else "m_layout::row_major"
        b_enum = f"m_layout::{b_layout}" if b_layout != "any" else "m_layout::row_major"
        c_enum = f"m_layout::{c_layout}" if c_layout != "any" else "m_layout::row_major"

        code += f"        {{{{{M}, {N}, {K}, {type_group}, {a_enum}, {b_enum}, {c_enum}}}, {config_idx}}}"
        code += "," if i < len(sorted_configs) - 1 else ""
        code += f" // {M}x{N}x{K}, group={type_group}, A={a_layout}, B={b_layout}, C={c_layout}\n"

    code += """    }};

    // Find closest configuration for a given type_group and layout.
    // Strategy: closest K first, then closest M,N. Falls back to any layout if needed.
    constexpr size_t find_closest_config(size_t m, size_t n, size_t k,
                                         size_t type_group,
                                         m_layout layout_a,
                                         m_layout layout_b,
                                         m_layout layout_c)
    {
        if(sorted_config_map.empty())
        {
            return DEFAULT_CONFIG_IDX;
        }

        auto mn_distance = [](size_t m1, size_t n1, size_t m2, size_t n2) -> double
        {
            double dm = static_cast<double>(m1) - static_cast<double>(m2);
            double dn = static_cast<double>(n1) - static_cast<double>(n2);
            return dm * dm + dn * dn;
        };

        auto search = [&](size_t tg) -> size_t
        {
            size_t closest_k = 0;
            size_t min_k_diff = std::numeric_limits<size_t>::max();
            bool found = false;

            for(size_t i = 0; i < sorted_config_map.size(); ++i)
            {
                const auto& key = sorted_config_map[i].first;
                if(key.type_group == tg &&
                   key.layout_a == layout_a && key.layout_b == layout_b && key.layout_c == layout_c)
                {
                    found = true;
                    size_t kd = (key.k > k) ? (key.k - k) : (k - key.k);
                    if(kd < min_k_diff) { min_k_diff = kd; closest_k = key.k; }
                }
            }

            if(!found) return DEFAULT_CONFIG_IDX;

            double min_dist = std::numeric_limits<double>::max();
            size_t best = DEFAULT_CONFIG_IDX;
            for(size_t i = 0; i < sorted_config_map.size(); ++i)
            {
                const auto& key = sorted_config_map[i].first;
                if(key.type_group == tg && key.k == closest_k &&
                   key.layout_a == layout_a && key.layout_b == layout_b && key.layout_c == layout_c)
                {
                    double d = mn_distance(m, n, key.m, key.n);
                    if(d < min_dist) { min_dist = d; best = i; }
                }
            }
            return sorted_config_map[best].second;
        };

        size_t result = search(type_group);
        // Fall back to group 0 if no entries exist for requested type_group
        if(result == DEFAULT_CONFIG_IDX && type_group != 0)
        {
            result = search(0);
        }
        return result;
    }

    // Find the best configuration for a given type_group using binary search.
    // Falls back to type_group 0 (f16/bf16) when no dedicated config exists.
    constexpr size_t find_best_config(size_t m, size_t n, size_t k,
                                      size_t type_group,
                                      m_layout layout_a,
                                      m_layout layout_b,
                                      m_layout layout_c)
    {
        auto search_exact = [&](size_t tg) -> size_t
        {
            config_key target{m, n, k, tg, layout_a, layout_b, layout_c};
            auto it = std::lower_bound(
                sorted_config_map.begin(), sorted_config_map.end(),
                std::make_pair(target, size_t(0)),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            if(it != sorted_config_map.end() && it->first == target)
            {
                return it->second;
            }
            return DEFAULT_CONFIG_IDX;
        };

        // Exact match for requested type_group
        size_t result = search_exact(type_group);
        if(result != DEFAULT_CONFIG_IDX) return result;

        // Exact match for fallback group 0
        if(type_group != 0)
        {
            result = search_exact(0);
            if(result != DEFAULT_CONFIG_IDX) return result;
        }

        // Closest match
        return find_closest_config(m, n, k, type_group, layout_a, layout_b, layout_c);
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
                                       m_layout layout_b,
                                       size_t type_group = 0)
{
    const size_t config_idx = detail::find_best_config(m, n, k, type_group,
                                                        layout_a, layout_b, layout_c);
    const auto& config = detail::kernel_configs[config_idx];
    return gemm_params{
        std::get<0>(config),
        std::get<1>(config),
        std::get<2>(config),
        std::get<3>(config),
        std::get<4>(config),
        std::get<5>(config)
    };
}

} // namespace rocm_wmma_gemm

#endif // ROCM_WMMA_GEMM_CONFIG_GENERATED_HPP"""

    with open(output_file, 'w') as f:
        f.write(code)


def generate_kernel_sources(f16_config_file, output_dir, f32_config_file=None):
    """Generate one source file per (config, layout) containing both aligned and unaligned variants."""
    all_confs, _ = load_configs(f16_config_file, f32_config_file)
    # Flatten to just (conf, _) and deduplicate on config+layout (type_group doesn't
    # affect kernel instantiation — both groups share the same kernel_gemm_impl indices)
    config = {'configurations': [c for c, _ in all_confs]}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    type_pairs = [
        ('half', 'half'),
        ('float', 'half'),
        ('__hip_bfloat16', '__hip_bfloat16'),
        ('float', '__hip_bfloat16')
    ]

    seen_combinations = set()
    file_list = []
    file_index = 0  # Each file covers one (config, layout); aligned=file_index, unaligned=file_index+1

    for conf in config['configurations']:
        cfg = conf['config']
        layout = conf['layout']
        range_info = conf['range']

        config_key = (
            cfg['warps_m'], cfg['warps_n'],
            cfg['warp_tile_m'], cfg['warp_tile_n'],
            cfg['swizzle'],
            cfg['bits'],
            layout['A'], layout['B'], layout['C']
        )

        if config_key in seen_combinations:
            continue
        seen_combinations.add(config_key)

        wm, wn, wtm, wtn, swizzle_val, bits = config_key[:6]
        layout_a, layout_b, layout_c = config_key[6:]
        layout_str = f"{layout_a[0]}{layout_b[0]}{layout_c[0]}"

        # One file holds both aligned and unaligned; indices are file_index (aligned)
        # and file_index+1 (unaligned) so the lookup table mapping is unchanged.
        aligned_idx   = file_index
        unaligned_idx = file_index + 1

        filename = f"kernel_inst_{file_index}.cpp"
        filepath = output_path / filename
        file_list.append(filename)

        code = f"""// Auto-generated kernel instantiation file - DO NOT EDIT
// Config: warps({wm},{wn}) tiles({wtm},{wtn}) swizzle({swizzle_val}) bits({bits})
// Layout: A={layout_a}, B={layout_b}, C={layout_c}
// Size hint: {range_info['M']}x{range_info['N']}x{range_info['K']}

#include <rocm_wmma_gemm/kernel/kernel.hpp>

namespace rocm_wmma_gemm
{{

"""
        for t_type, u_type in type_pairs:
            t_name = t_type.replace('__hip_bfloat16', 'bf16').replace('half', 'f16').replace('float', 'f32')
            u_name = u_type.replace('__hip_bfloat16', 'bf16').replace('half', 'f16').replace('float', 'f32')

            code += f"""extern "C" void* get_kernel_inst_{aligned_idx}_{t_name}_{u_name}_{layout_str}_aligned() {{
    return (void*)&kernel_gemm_impl<{t_type}, {u_type},
        m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
        {wm}, {wn}, {wtm}, {wtn}, {swizzle_val}, {bits}, 1>::run;
}}

extern "C" void* get_kernel_inst_{unaligned_idx}_{t_name}_{u_name}_{layout_str}_unaligned() {{
    return (void*)&kernel_gemm_impl<{t_type}, {u_type},
        m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
        {wm}, {wn}, {wtm}, {wtn}, {swizzle_val}, {bits}, 0>::run;
}}

"""

        code += """} // namespace rocm_wmma_gemm
"""
        with open(filepath, 'w') as f:
            f.write(code)

        file_index += 2  # Reserve two logical indices per file (aligned + unaligned)

    filelist_path = output_path / "kernel_sources.txt"
    with open(filelist_path, 'w') as f:
        for filename in file_list:
            f.write(f"{filename}\n")

    cmake_file = output_path / "kernel_sources.cmake"
    with open(cmake_file, 'w') as f:
        f.write("# Auto-generated list of kernel source files\n")
        f.write("set(KERNEL_INST_SOURCES\n")
        for filename in file_list:
            f.write(f"    {output_path}/{filename}\n")
        f.write(")\n")

    print(f"Generated {len(file_list)} kernel source files (each contains aligned + unaligned variants)")

    generate_kernel_lookup(f16_config_file, output_path, file_list, f32_config_file)

    return file_list


def generate_kernel_lookup(f16_config_file, output_dir, file_list, f32_config_file=None):
    """Generate the kernel lookup implementation with static [config][type][layout][alignment] table"""
    all_confs, unique_configs = load_configs(f16_config_file, f32_config_file)
    # unique_configs already built by load_configs; rebuild for local use
    unique_configs_set = set()
    for conf, _ in all_confs:
        cfg = conf['config']
        unique_configs_set.add((cfg['warps_m'], cfg['warps_n'],
                                cfg['warp_tile_m'], cfg['warp_tile_n'],
                                cfg['swizzle'],
                                cfg['bits']))

    default_configs = [(4, 4, 4, 4, 8, 256)]
    for dc in default_configs:
        if dc not in unique_configs_set:
            unique_configs_set.add(dc)

    unique_configs = sorted(list(unique_configs_set))
    num_configs = len(unique_configs)

    type_pairs = [
        ('half', 'half', 'f16', 'f16'),
        ('float', 'half', 'f32', 'f16'),
        ('__hip_bfloat16', '__hip_bfloat16', 'bf16', 'bf16'),
        ('float', '__hip_bfloat16', 'f32', 'bf16')
    ]

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

    # Build mapping from (config_tuple, layout_tuple) -> (aligned_file_idx, unaligned_file_idx)
    config_layout_to_files = {}
    seen_combinations = set()
    file_index = 0

    for conf, _ in all_confs:
        cfg = conf['config']
        layout = conf['layout']

        config_key = (
            cfg['warps_m'], cfg['warps_n'],
            cfg['warp_tile_m'], cfg['warp_tile_n'],
            cfg['swizzle'], cfg['bits'],
            layout['A'], layout['B'], layout['C']
        )

        if config_key in seen_combinations:
            continue
        seen_combinations.add(config_key)

        config_tuple = config_key[:6]
        layout_tuple = (layout['A'], layout['B'], layout['C'])
        config_layout_to_files[(config_tuple, layout_tuple)] = (file_index, file_index + 1)
        file_index += 2

    filepath = output_dir / "kernel_lookup.cpp"

    code = """// Auto-generated kernel lookup - DO NOT EDIT
#include <rocm_wmma_gemm/kernel/common.hpp>
#include <cstddef>

namespace rocm_wmma_gemm
{

"""

    # Forward declare all getters
    for (config_tuple, layout_tuple), (aligned_idx, unaligned_idx) in config_layout_to_files.items():
        layout_str = next(ls for la, lb, lc, ls in layouts if (la, lb, lc) == layout_tuple)
        for _, _, t_name, u_name in type_pairs:
            code += f'extern "C" void* get_kernel_inst_{aligned_idx}_{t_name}_{u_name}_{layout_str}_aligned();\n'
            code += f'extern "C" void* get_kernel_inst_{unaligned_idx}_{t_name}_{u_name}_{layout_str}_unaligned();\n'

    code += f"""
// Static kernel lookup table: [config_idx][type_idx][layout_idx][alignment_idx]
// config_idx: 0 to {num_configs-1}
// type_idx: 0=half-half, 1=float-half, 2=bf16-bf16, 3=float-bf16
// layout_idx: 0=rrr, 1=rrc, 2=rcr, 3=rcc, 4=crr, 5=crc, 6=ccr, 7=ccc
// alignment_idx: 0=unaligned, 1=aligned
static void* kernel_table[{num_configs}][4][8][2] = {{
"""

    for config_idx, config_tuple in enumerate(unique_configs):
        code += f"    // Config {config_idx}: warps({config_tuple[0]},{config_tuple[1]}) tiles({config_tuple[2]},{config_tuple[3]}) swizzle({config_tuple[4]}) bits({config_tuple[5]})\n"
        code += "    {\n"

        for type_idx, (_, _, t_name, u_name) in enumerate(type_pairs):
            code += "        {"

            for layout_idx, (layout_a, layout_b, layout_c, layout_str) in enumerate(layouts):
                layout_tuple = (layout_a, layout_b, layout_c)
                key = (config_tuple, layout_tuple)

                code += "{"
                if key in config_layout_to_files:
                    aligned_idx, unaligned_idx = config_layout_to_files[key]
                    code += f"get_kernel_inst_{unaligned_idx}_{t_name}_{u_name}_{layout_str}_unaligned(), "
                    code += f"get_kernel_inst_{aligned_idx}_{t_name}_{u_name}_{layout_str}_aligned()"
                else:
                    code += "nullptr, nullptr"
                code += "}"

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

void* lookup_kernel(size_t config_idx, size_t type_idx, size_t layout_idx, size_t alignment_idx)
{
    return kernel_table[config_idx][type_idx][layout_idx][alignment_idx];
}

} // namespace rocm_wmma_gemm
"""

    with open(filepath, 'w') as f:
        f.write(code)

    print(f"Generated kernel lookup table in {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Generate WMMA GEMM configuration header and kernel sources')
    parser.add_argument('--f16-config', type=str, required=True,
                        help='JSON config file for f16/bf16 type groups')
    parser.add_argument('--f32-config', type=str, default=None,
                        help='JSON config file for f32 accumulator type groups (optional)')
    parser.add_argument('output_file', type=str, help='Output header file')
    parser.add_argument('--kernel-dir', type=str, help='Output directory for kernel source files')

    args = parser.parse_args()

    generate_config_header(args.f16_config, args.output_file, args.f32_config)

    if args.kernel_dir:
        generate_kernel_sources(args.f16_config, args.kernel_dir, args.f32_config)


if __name__ == '__main__':
    main()
