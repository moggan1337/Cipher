"""
Benchmark Suite for Cipher FHE Framework

This module provides comprehensive benchmarking for:
- Key generation time
- Encryption/decryption throughput
- Homomorphic operation performance
- Neural network inference speed
- Memory usage

Usage:
    python -m benchmarks.benchmark
"""

import time
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.params import CKKSParameters, SecurityLevel
from src.core.context import FHEContext
from src.core.ckks import Ciphertext
from src.ml.encrypted_model import EncryptedLinear, EncryptedReLU, EncryptedSequential
from src.ml.approximation import PolynomialApproximator


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    time_seconds: float
    throughput: float
    metadata: Dict = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.time_seconds:.4f}s ({self.throughput:.2f} ops/s)"


class BenchmarkSuite:
    """Comprehensive benchmark suite."""
    
    def __init__(self, poly_degree: int = 8192):
        self.poly_degree = poly_degree
        self.results: List[BenchmarkResult] = []
        self.context: FHEContext = None
    
    def setup(self):
        """Initialize context and keys."""
        print(f"\n{'='*60}")
        print(f"Benchmark Suite - Cipher FHE Framework")
        print(f"Polynomial Degree: {self.poly_degree}")
        print(f"{'='*60}\n")
        
        # Create context with small parameters for speed
        self.context = FHEContext.default(
            security_level=128,
            poly_degree=self.poly_degree,
            enable_bootstrapping=False,
        )
        
        print("Generating keys (one-time cost)...")
        start = time.time()
        self.context.setup_keys()
        keygen_time = time.time() - start
        print(f"Key generation: {keygen_time:.2f}s\n")
    
    def benchmark_key_generation(self) -> BenchmarkResult:
        """Benchmark key generation."""
        print("\n--- Key Generation Benchmark ---")
        
        times = []
        for _ in range(3):
            start = time.time()
            self.context.setup_keys()  # Regenerate
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        result = BenchmarkResult(
            name="Key Generation",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
            metadata={"poly_degree": self.poly_degree},
        )
        self.results.append(result)
        print(f"  Average time: {avg_time:.3f}s")
        return result
    
    def benchmark_encryption(self, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark encryption throughput."""
        print("\n--- Encryption Benchmark ---")
        
        # Generate test data
        values = np.random.randn(self.context.params.slot_count)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            ct = self.context.encrypt(values)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        result = BenchmarkResult(
            name="Encryption",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
            metadata={
                "slot_count": self.context.params.slot_count,
                "std_dev": std_time,
            },
        )
        self.results.append(result)
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {1.0/avg_time:.1f} encryptions/s")
        return result
    
    def benchmark_decryption(self, num_iterations: int = 100) -> BenchmarkResult:
        """Benchmark decryption throughput."""
        print("\n--- Decryption Benchmark ---")
        
        # Pre-generate ciphertext
        values = np.random.randn(self.context.params.slot_count)
        ct = self.context.encrypt(values)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.context.decrypt(ct)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        result = BenchmarkResult(
            name="Decryption",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
            metadata={"slot_count": self.context.params.slot_count},
        )
        self.results.append(result)
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {1.0/avg_time:.1f} decryptions/s")
        return result
    
    def benchmark_addition(self, num_iterations: int = 1000) -> BenchmarkResult:
        """Benchmark homomorphic addition."""
        print("\n--- Addition Benchmark ---")
        
        values = np.random.randn(self.context.params.slot_count)
        ct1 = self.context.encrypt(values)
        ct2 = self.context.encrypt(values)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.context.add(ct1, ct2)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        result = BenchmarkResult(
            name="Addition",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
        )
        self.results.append(result)
        print(f"  Average time: {avg_time*1000:.4f}ms")
        print(f"  Throughput: {1.0/avg_time:.1f} additions/s")
        return result
    
    def benchmark_multiplication(self, num_iterations: int = 500) -> BenchmarkResult:
        """Benchmark homomorphic multiplication."""
        print("\n--- Multiplication Benchmark ---")
        
        values = np.random.randn(self.context.params.slot_count)
        ct1 = self.context.encrypt(values)
        ct2 = self.context.encrypt(values)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.context.multiply(ct1, ct2)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        result = BenchmarkResult(
            name="Multiplication",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
        )
        self.results.append(result)
        print(f"  Average time: {avg_time*1000:.4f}ms")
        print(f"  Throughput: {1.0/avg_time:.1f} multiplications/s")
        return result
    
    def benchmark_square(self, num_iterations: int = 500) -> BenchmarkResult:
        """Benchmark squaring."""
        print("\n--- Squaring Benchmark ---")
        
        values = np.random.randn(self.context.params.slot_count)
        ct = self.context.encrypt(values)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.context.square(ct)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        result = BenchmarkResult(
            name="Square",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
        )
        self.results.append(result)
        print(f"  Average time: {avg_time*1000:.4f}ms")
        print(f"  Throughput: {1.0/avg_time:.1f} squarings/s")
        return result
    
    def benchmark_polynomial_evaluation(
        self, 
        degree: int = 5,
        num_iterations: int = 100
    ) -> BenchmarkResult:
        """Benchmark polynomial evaluation (activation approximation)."""
        print(f"\n--- Polynomial Evaluation (degree={degree}) Benchmark ---")
        
        values = np.random.randn(self.context.params.slot_count) * 0.5
        ct = self.context.encrypt(values)
        
        approx = PolynomialApproximator(self.context)
        poly = approx.get_relu(degree)
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = approx.evaluate(ct, poly)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        result = BenchmarkResult(
            name=f"PolyEval_d{degree}",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
            metadata={"polynomial_degree": degree},
        )
        self.results.append(result)
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Throughput: {1.0/avg_time:.1f} evaluations/s")
        return result
    
    def benchmark_neural_network_inference(
        self,
        architecture: List[int] = [784, 256, 128, 10],
        num_iterations: int = 10,
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark neural network inference."""
        print(f"\n--- Neural Network Inference Benchmark ---")
        print(f"Architecture: {' -> '.join(map(str, architecture))}")
        
        # Create model
        layers = []
        for i in range(len(architecture) - 1):
            in_f, out_f = architecture[i], architecture[i+1]
            layers.append(EncryptedLinear(in_f, out_f))
            if i < len(architecture) - 2:
                layers.append(EncryptedReLU(degree=3))
        
        model = EncryptedSequential(layers)
        
        # Set random weights
        for layer in model.layers:
            if isinstance(layer, EncryptedLinear):
                w = np.random.randn(layer.out_features, layer.in_features) * 0.01
                layer.set_weights(w)
        
        # Generate input
        x = np.random.randn(architecture[0]) * 0.1
        ct_x = self.context.encrypt(x)
        
        # Warm up
        _ = model.forward(ct_x, self.context)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = model.forward(ct_x, self.context)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        # Estimate depth usage
        from src.ml.encrypted_model import estimate_inference_depth
        depth = estimate_inference_depth(model, activation_degree=3)
        
        result = BenchmarkResult(
            name="NN_Inference",
            time_seconds=avg_time,
            throughput=1.0 / avg_time,
            metadata={
                "architecture": architecture,
                "total_params": sum(a*b for a, b in zip(architecture[:-1], architecture[1:])),
                "estimated_depth": depth,
            },
        )
        self.results.append(result)
        
        print(f"  Average inference time: {avg_time*1000:.1f}ms")
        print(f"  Throughput: {1.0/avg_time:.2f} inferences/s")
        print(f"  Estimated multiplicative depth: {depth}")
        
        return {"inference": result}
    
    def benchmark_matrix_multiplication(
        self,
        matrix_shape: Tuple[int, int] = (256, 784),
        num_iterations: int = 50,
    ) -> BenchmarkResult:
        """Benchmark matrix-vector multiplication."""
        print(f"\n--- Matrix Multiplication Benchmark ---")
        print(f"Matrix shape: {matrix_shape}")
        
        # Generate data
        matrix = np.random.randn(*matrix_shape) * 0.01
        vector = np.random.randn(matrix_shape[1]) * 0.1
        
        ct_vector = self.context.encrypt(vector)
        
        # Warm up
        _ = self.context.matrix_multiply(ct_vector, matrix)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.context.matrix_multiply(ct_vector, matrix)
            times.append(time.time() - start)
        
        avg_time = statistics.mean(times)
        
        flops = 2 * matrix_shape[0] * matrix_shape[1]  # Multiply-adds
        
        result = BenchmarkResult(
            name="MatMult",
            time_seconds=avg_time,
            throughput=flops / avg_time / 1e9,  # GFLOPS
            metadata={
                "matrix_shape": matrix_shape,
                "flops": flops,
            },
        )
        self.results.append(result)
        
        print(f"  Average time: {avg_time*1000:.1f}ms")
        print(f"  Throughput: {flops/avg_time/1e9:.3f} GFLOPS")
        return result
    
    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        self.setup()
        
        # Core operations
        self.benchmark_encryption(50)
        self.benchmark_decryption(50)
        self.benchmark_addition(500)
        self.benchmark_multiplication(200)
        self.benchmark_square(200)
        
        # Polynomial evaluation
        for degree in [3, 5, 7]:
            self.benchmark_polynomial_evaluation(degree, 50)
        
        # Neural network
        self.benchmark_neural_network_inference([784, 256, 128, 10], 5)
        
        # Matrix multiplication
        self.benchmark_matrix_multiplication((256, 784), 20)
        
        return self.results
    
    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for result in self.results:
            if "GFLOPS" in str(result.throughput):
                print(f"{result.name:25s}: {result.time_seconds*1000:8.1f}ms  {result.throughput:8.3f} GFLOPS")
            elif result.throughput < 10:
                print(f"{result.name:25s}: {result.time_seconds*1000:8.1f}ms  {result.throughput:8.2f} ops/s")
            else:
                print(f"{result.name:25s}: {result.time_seconds*1000:8.2f}ms  {result.throughput:8.0f} ops/s")


def compare_poly_degrees():
    """Compare performance across different polynomial degrees."""
    print("\n" + "="*60)
    print("POLYNOMIAL DEGREE COMPARISON")
    print("="*60)
    
    degrees = [1024, 2048, 4096]
    results = {}
    
    for degree in degrees:
        print(f"\n--- Testing N={degree} ---")
        suite = BenchmarkSuite(poly_degree=degree)
        suite.setup()
        
        enc_result = suite.benchmark_encryption(20)
        mul_result = suite.benchmark_multiplication(50)
        
        results[degree] = {
            "encryption_ms": enc_result.time_seconds * 1000,
            "multiplication_ms": mul_result.time_seconds * 1000,
            "slots": degree // 2,
        }
    
    print("\n" + "-"*60)
    print(f"{'Degree':<10} {'Slots':<10} {'Encrypt (ms)':<15} {'Multiply (ms)':<15}")
    print("-"*60)
    for degree, data in results.items():
        print(f"{degree:<10} {data['slots']:<10} {data['encryption_ms']:<15.2f} {data['multiplication_ms']:<15.2f}")


def main():
    """Main benchmark entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cipher FHE Benchmark Suite")
    parser.add_argument("--degree", type=int, default=4096, help="Polynomial degree")
    parser.add_argument("--compare", action="store_true", help="Compare across degrees")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_poly_degrees()
    else:
        suite = BenchmarkSuite(poly_degree=args.degree)
        
        if args.quick:
            suite.setup()
            suite.benchmark_encryption(10)
            suite.benchmark_multiplication(20)
        else:
            suite.run_all()
        
        suite.print_summary()


if __name__ == "__main__":
    main()
