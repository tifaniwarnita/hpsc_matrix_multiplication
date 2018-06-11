#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>

#define BLOCK 16

__global__ void matmul(float *A, float *B, float *C, int M, int N, int K) {
	// Shared memory
	__shared__ float s_A[BLOCK][BLOCK];
	__shared__ float s_B[BLOCK][BLOCK];

	int a_begin = N * BLOCK * blockIdx.y; // N * blockDim.y * blockIdx.y
	int a_end = a_begin + N;
	int a_step = BLOCK; // blockDim.x

	int b_begin = BLOCK * blockIdx.x; // blockDim.y * blockIdx.x
	int b_step = BLOCK * K; // blockDim.y * K

	int a = a_begin;
	int b = b_begin;

	int a_th = N * threadIdx.y + threadIdx.x;
	int b_th = K * threadIdx.y + threadIdx.x;

	float sum = 0;

	while (a < a_end) {
		// Copy to shared memory
		__syncthreads();
		s_A[threadIdx.y][threadIdx.x] = A[a + a_th];
		s_B[threadIdx.y][threadIdx.x] = B[b + b_th];
		__syncthreads();

		// Multiply
	#pragma unroll
		for (int i=0; i<BLOCK; i++) {
			sum += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
		}

		a += a_step;
		b += b_step;
	}

	if (blockIdx.y * BLOCK + threadIdx.y >= M) return;
	if (blockIdx.x * BLOCK + threadIdx.x >= K) return;

	int c_idx = \
		K * BLOCK * blockIdx.y + \
		BLOCK * blockIdx.x + \
		K * threadIdx.y + \
		threadIdx.x;

	C[c_idx] = sum;
}
 
int main(int argc, char **argv) {
	int M, N, K;
	switch(argc) {
		case 2: M = atoi(argv[1]);
				N = M;
				K = M;
				break;
		case 4:	M = atoi(argv[1]);
				N = atoi(argv[2]);
				K = atoi(argv[3]);
				break;
		default: printf("Invalid number of parameters\n");
				 return 1;
	}

	// Host allocation
	float *h_A = new float [M*N];
	float *h_B = new float [N*K];
	float *h_C = new float [M*K];

	for (int i=0; i<M*N; i++) {
		h_A[i] = drand48();
	}
	for (int i=0; i<N*K; i++) {
		h_B[i] = drand48();
	}
	for (int i=0; i<M*K; i++) {
		h_C[i] = 0;
	}

	// Device allocation
	float *d_A, *d_B, *d_C;
	int mem_A = M * N * sizeof(float);
	int mem_B = N * K * sizeof(float);
	int mem_C = M * K * sizeof(float);
	cudaMalloc((void **) &d_A, mem_A);
	cudaMalloc((void **) &d_B, mem_B);
	cudaMalloc((void **) &d_C, mem_C);

	// Copy from Host to Device
	cudaMemcpy(d_A, h_A, mem_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, mem_B, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_C, cudaMemcpyHostToDevice);

	dim3 grid((K/BLOCK)+(K%BLOCK!=0),(M/BLOCK)+(M%BLOCK!=0)); // number of blocks
	dim3 block(BLOCK,BLOCK); // threads per block

	// CUDA
	struct timeval tic, toc;
	gettimeofday(&tic, NULL);
	matmul<<<grid,block>>>(d_A, d_B, d_C, M, N, K);
	cudaDeviceSynchronize();
	gettimeofday(&toc, NULL);
	double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
	printf("[%dx%dx%d]\n", M, N, K);
	printf("CUDA  : %lfs (%lf GFlops)\n", time, 2.*M*N*K/time/1e9);

	// Copy Result from　Device to Host
	cudaMemcpy(h_C, d_C, mem_C, cudaMemcpyDeviceToHost);

	// CPU
	gettimeofday(&tic, NULL);
#pragma omp parallel for
	for (int i=0; i<M; ++i) {
		for(int j=0; j<K; j++) {
			for (int k=0; k<N; k++) {
				h_C[K*i+j] -= h_A[N*i+k] * h_B[K*k+j];
			}
		}
	}
	gettimeofday(&toc, NULL);
	time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
	printf("CPU   : %lfs (%lf GFlops)\n", time, 2.*M*N*K/time/1e9);

	// Calculate error
	float err = 0;
	for (int i=0; i<M*K; ++i) {
		err += fabs(h_C[i]);
	}
	printf("Error : %f\n",err/M/K);

	// Free memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
