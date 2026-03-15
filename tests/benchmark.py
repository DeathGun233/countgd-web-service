import requests
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8000/predict"
TEST_IMAGE = "/home/Yjh/fish_count_deploy/tests/middle3-0-1.jpg"

def single_request():
    """发送单个请求"""
    start = time.time()
    
    with open(TEST_IMAGE, 'rb') as f:
        files = {'file': f}
        response = requests.post(API_URL, files=files)
    
    elapsed = (time.time() - start) * 1000  # ms
    
    if response.status_code == 200:
        return elapsed, response.json()
    else:
        return None, None

def benchmark_sequential(num_requests=50):
    """顺序请求测试"""
    print(f"\n{'='*60}")
    print(f"顺序请求测试 ({num_requests} 次)")
    print(f"{'='*60}")
    
    times = []
    
    for i in range(num_requests):
        elapsed, result = single_request()
        if elapsed:
            times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"  已完成 {i+1}/{num_requests} 次请求")
    
    # 统计
    if times:
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        qps = 1000 / avg_time
        
        print(f"\n结果:")
        print(f"  总请求数: {len(times)}")
        print(f"  平均响应时间: {avg_time:.2f} ms")
        print(f"  中位数响应时间: {median_time:.2f} ms")
        print(f"  最快响应: {min_time:.2f} ms")
        print(f"  最慢响应: {max_time:.2f} ms")
        print(f"  QPS: {qps:.2f}")

def benchmark_concurrent(num_requests=100, concurrency=10):
    """并发请求测试"""
    print(f"\n{'='*60}")
    print(f"并发请求测试 ({num_requests} 次, 并发={concurrency})")
    print(f"{'='*60}")
    
    times = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        
        for i, future in enumerate(as_completed(futures)):
            elapsed, result = future.result()
            if elapsed:
                times.append(elapsed)
            
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i+1}/{num_requests} 次请求")
    
    total_time = time.time() - start_time
    
    # 统计
    if times:
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        qps = num_requests / total_time
        
        print(f"\n结果:")
        print(f"  总请求数: {len(times)}")
        print(f"  总耗时: {total_time:.2f} s")
        print(f"  平均响应时间: {avg_time:.2f} ms")
        print(f"  中位数响应时间: {median_time:.2f} ms")
        print(f"  最快响应: {min_time:.2f} ms")
        print(f"  最慢响应: {max_time:.2f} ms")
        print(f"  QPS: {qps:.2f}")

def main():
    print("CountGD API 性能测试")
    print(f"API地址: {API_URL}")
    print(f"测试图片: {TEST_IMAGE}")
    
    # 1. 顺序测试
    benchmark_sequential(num_requests=50)
    
    # 2. 并发测试
    benchmark_concurrent(num_requests=100, concurrency=10)
    
    print("\n测试完成！")

if __name__ == '__main__':
    main()