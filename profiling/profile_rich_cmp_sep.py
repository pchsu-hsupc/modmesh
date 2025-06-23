import functools
import gc
import pprint
import time

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


def profile_rich_cmp(pow, dtype_name="uint8", it=50):
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

    def generate_test_data():
        """動態生成測試數據"""
        if dtype_name == "bool":
            test_data1 = np.random.choice([True, False], size=N).astype(dtype)
            test_data2 = np.random.choice([True, False], size=N).astype(dtype)
        elif dtype_name.startswith("float"):
            test_data1 = (np.random.rand(N) * 1000).astype(dtype)
            test_data2 = (np.random.rand(N) * 1000).astype(dtype)
        else:
            test_data1 = np.arange(0, N, dtype=dtype)
            test_data2 = np.arange(0, N, dtype=dtype)
            np.random.shuffle(test_data1)
            np.random.shuffle(test_data2)
        return test_data1, test_data2

    modmesh.call_profiler.reset()
    
    # 預熱 - numpy
    warmup_count = min(5, it // 10)
    print(f"Warming up numpy operations for {dtype_name}... ({warmup_count} iterations)")
    for i in range(warmup_count):
        test_data1, test_data2 = generate_test_data()
        profile_rich_cmp_np(test_data1, test_data2)
        # 立即釋放記憶體
        del test_data1, test_data2
    
    # 清理和穩定
    gc.collect()
    time.sleep(0.1)
    
    # 正式測試 - numpy 獨立迴圈
    print(f"Testing numpy operations for {dtype_name}... ({it} iterations)")
    for i in range(it):
        test_data1, test_data2 = generate_test_data()
        profile_rich_cmp_np(test_data1, test_data2)
        del test_data1, test_data2
        
        # 每10次迭代進行一次垃圾回收
        if (i + 1) % 10 == 0:
            gc.collect()
    
    # 預熱 - SimpleArray
    print(f"Warming up SimpleArray operations for {dtype_name}... ({warmup_count} iterations)")
    for i in range(warmup_count):
        test_data1, test_data2 = generate_test_data()
        test_sa1 = make_container(test_data1)
        test_sa2 = make_container(test_data2)
        profile_rich_cmp_sa(test_sa1, test_sa2)
        del test_data1, test_data2, test_sa1, test_sa2
    
    # 清理和穩定
    gc.collect()
    time.sleep(0.1)
    
    # 正式測試 - SimpleArray 獨立迴圈
    print(f"Testing SimpleArray operations for {dtype_name}... ({it} iterations)")
    for i in range(it):
        test_data1, test_data2 = generate_test_data()
        test_sa1 = make_container(test_data1)
        test_sa2 = make_container(test_data2)
        profile_rich_cmp_sa(test_sa1, test_sa2)
        del test_data1, test_data2, test_sa1, test_sa2
        
        # 每10次迭代進行一次垃圾回收
        if (i + 1) % 10 == 0:
            gc.collect()

    res = modmesh.call_profiler.result()["children"]

    print(f"## N = {2 ** (pow % 10)}{ORDER} type: {dtype_name} (iterations: {it})\n")
    out = {}
    stats = {}
    
    for r in res:
        name = r["name"].replace("profile_rich_cmp_", "")
        total_time = r["total_time"]
        count = r["count"]
        avg_time = total_time / count
        out[name] = avg_time
        stats[name] = {
            'total': total_time,
            'count': count,
            'avg': avg_time,
            'per_element': avg_time / N,
            'std_dev': r.get('std_dev', 0)  # 如果有標準差數據
        }

    def print_row(*cols):
        print(str.format("| {:10s} | {:15s} | {:15s} | {:15s} | {:15s} |",
                         *(cols[0:5])))

    print_row('func', 'per call (ms)', 'per elem (ns)', 'cmp to np', 'cmp to sa')
    print_row('-' * 10, '-' * 15, '-' * 15, '-' * 15, '-' * 15)
    
    if "np" in out and "sa" in out:
        npbase = out["np"]
        sabase = out["sa"]
        for k, v in out.items():
            per_elem_ns = stats[k]['per_element'] * 1e9
            print_row(f"{k:8s}", f"{v:.3E}", f"{per_elem_ns:.3f}", 
                     f"{v/npbase:.3f}", f"{v/sabase:.3f}")
    
        print(f"Verification: np_count={stats['np']['count']}, sa_count={stats['sa']['count']}")
        print(f"Performance ratio (SA/NP): {sabase/npbase:.3f}")
    else:
        print("Warning: Missing numpy or SimpleArray results")
    
    # 最終清理
    gc.collect()
    print()


def main():
    pprint.pp(np.show_runtime())
    initial_pow = 7
    it = 10
    
    # Test different data types
    dtypes_to_test = [
        "bool", "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64", 
        "float32", "float64"
    ]

    for dtype_name in dtypes_to_test:
        print(f"\n=== Testing {dtype_name} ===")
        pow = initial_pow
        for test_round in range(it):
            print(f"Round {test_round + 1}/{it}, pow={pow}, N={2**pow}")
            profile_rich_cmp(pow, dtype_name, it=50)
            pow = pow + 3
            
            # 在每個測試輪次後強制垃圾回收
            gc.collect()
            time.sleep(0.1)  # 給系統一點時間清理記憶體


if __name__ == "__main__":
    main()