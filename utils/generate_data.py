"""
Generates random data for matrix multiplication.
"""

import os
import numpy as np

def generate_matmul_test_data(dir, M, N, K):
    os.makedirs(dir, exist_ok=True)
    a = np.random.randn(M * K).astype(np.float16).reshape((M, K))
    b = np.random.randn(M * K).astype(np.float16).reshape((K, N))
    a_path = os.path.join(dir, "a.csv")
    b_path = os.path.join(dir, "b.csv")
    c_path = os.path.join(dir, "c.csv")
    for x, path in [(a, a_path), (b, b_path)]:
        x.view(np.int16).ravel().tofile(path, sep=",")
    return a, b, a_path, b_path, c_path

generate_matmul_test_data("data", 16, 16, 16)