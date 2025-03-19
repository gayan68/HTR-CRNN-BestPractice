import torch

device = 'cuda:0'  # Get the device from config

if torch.cuda.is_available():
    print(f"CUDA Available: {device}")
else:
    print("CPU")


print(torch.cuda.device_count())
print(torch.cuda.current_device())


# Create two random matrices
A = torch.rand(3, 3)  # 3x3 matrix
B = torch.rand(3, 3)  # 3x3 matrix

# Move matrices to GPU
A = A.to(device)
B = B.to(device)

# Perform matrix multiplication
C = torch.matmul(A, B)  # Or use C = A @ B for matrix multiplication

# Print the result
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nResulting Matrix C:")
print(C)
