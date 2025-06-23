# Copyright (c) 2025, Kuan-Hsien Lee <khlee870529@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import functools
import pprint

import numpy as np

import modmesh


def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ = modmesh.CallProfilerProbe(func.__name__)
        result = func(*args, **kwargs)
        return result
    return wrapper


def make_container(data):
    # Boolean type
    if np.isdtype(data.dtype, np.bool_):
        return modmesh.SimpleArrayBool(array=data)
    # Signed integer types
    elif np.isdtype(data.dtype, np.int8):
        return modmesh.SimpleArrayInt8(array=data)
    elif np.isdtype(data.dtype, np.int16):
        return modmesh.SimpleArrayInt16(array=data)
    elif np.isdtype(data.dtype, np.int32):
        return modmesh.SimpleArrayInt32(array=data)
    elif np.isdtype(data.dtype, np.int64):
        return modmesh.SimpleArrayInt64(array=data)
    # Unsigned integer types
    elif np.isdtype(data.dtype, np.uint8):
        return modmesh.SimpleArrayUint8(array=data)
    elif np.isdtype(data.dtype, np.uint16):
        return modmesh.SimpleArrayUint16(array=data)
    elif np.isdtype(data.dtype, np.uint32):
        return modmesh.SimpleArrayUint32(array=data)
    elif np.isdtype(data.dtype, np.uint64):
        return modmesh.SimpleArrayUint64(array=data)
    # Floating point types
    elif np.isdtype(data.dtype, np.float32):
        return modmesh.SimpleArrayFloat32(array=data)
    elif np.isdtype(data.dtype, np.float64):
        return modmesh.SimpleArrayFloat64(array=data)
    else:
        raise ValueError(f"Unsupported data type: {data.dtype}")


@profile_function
def profile_rich_cmp_np(narr, other):
    return narr < other


@profile_function
def profile_rich_cmp_sa(sarr, other):
    return sarr.lt(other)


def profile_rich_cmp(pow, dtype_name="uint8", it=10):
    N = 2 ** pow
    ORDER = ["", "K", "M", "G", "T"][pow // 10]
    
    # Map dtype names to numpy dtypes
    dtype_map = {
        "bool": np.bool_,
        "int8": np.int8, "int16": np.int16, "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8, "uint16": np.uint16, "uint32": np.uint32,
        "uint64": np.uint64,
        "float32": np.float32, "float64": np.float64
    }
    
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    
    dtype = dtype_map[dtype_name]

    modmesh.call_profiler.reset()
    for _ in range(it):
        if dtype_name == "bool":
            # For boolean, create random True/False arrays
            test_data1 = np.random.choice([True, False], size=N).astype(dtype)
            test_data2 = np.random.choice([True, False], size=N).astype(dtype)
        elif dtype_name.startswith("float"):
            # For float types, create random float arrays
            test_data1 = (np.random.rand(N) * 1000).astype(dtype)
            test_data2 = (np.random.rand(N) * 1000).astype(dtype)
        else:
            # For integer types, use the original approach
            test_data1 = np.arange(0, N, dtype=dtype)
            test_data2 = np.arange(0, N, dtype=dtype)
            np.random.shuffle(test_data1)
            np.random.shuffle(test_data2)
            
        test_sa1 = make_container(test_data1)
        test_sa2 = make_container(test_data2)
 
        profile_rich_cmp_np(test_data1, test_data2)
        profile_rich_cmp_sa(test_sa1, test_sa2)

    res = modmesh.call_profiler.result()["children"]

    print(f"## N = {2 ** (pow % 10)}{ORDER} type: {dtype_name}\n")
    out = {}
    for r in res:
        name = r["name"].replace("profile_rich_cmp_", "")
        time = r["total_time"] / r["count"]
        out[name] = time

    def print_row(*cols):
        print(str.format("| {:10s} | {:15s} | {:15s} |" " {:15s} |",
                         *(cols[0:4])))

    print_row('func', 'per call (ms)', 'cmp to np', 'cmp to sa')
    print_row('-' * 10, '-' * 15, '-' * 15, '-' * 15)
    npbase = out["np"]
    sabase = out["sa"]
    for k, v in out.items():
        print_row(f"{k:8s}", f"{v:.3E}", f"{v/npbase:.3f}", f"{v/sabase:.3f}")

    print()


def main():
    pprint.pp(np.show_runtime())
    initial_pow = 7
    it = 7
    
    # Test different data types
    dtypes_to_test = [
        "bool", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64"
    ]

    for dtype_name in dtypes_to_test:
        print(f"\n=== Testing {dtype_name} ===")
        pow = initial_pow  # Reset pow for each dtype
        for _ in range(it):
            profile_rich_cmp(pow, dtype_name)
            pow = pow + 3


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
