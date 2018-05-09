all:
	nvcc -Xcompiler "-O3 -fopenmp" cuda_baseline.cu -o cuda_baseline.x
	nvcc -Xcompiler "-O3 -fopenmp" cuda_optimized.cu -o cuda_optimized.x
clean:
	@rm -f cuda_baseline.x
	@rm -f cuda_optimized.x
