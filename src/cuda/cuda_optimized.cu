#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>

// #define DEBUG
#define BLOCK_SIZE 16

__global__ void
matmul(
    float *A, float *B, float *C,
    int height_A, int width_A,
    int height_B, int width_B) {
    // shared memory
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];

    // shorthand
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int a_begin = width_A * BLOCK_SIZE * by;
    int a_end = a_begin + width_A;
    int a_step = BLOCK_SIZE;

    int b_begin = BLOCK_SIZE * bx;
    int b_step = BLOCK_SIZE * width_B;

    float sum = 0.0;
    int a = a_begin;
    int b = b_begin;

    int a_temp = width_A * ty + tx;
    int b_temp = width_B * ty + tx;

    while (a < a_end) {
        // copy to shared memory
        __syncthreads();
        shared_A[ty][tx] = A[a + a_temp];
        shared_B[ty][tx] = B[b + b_temp];
        __syncthreads();

        // multiply
    #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += shared_A[ty][i] * shared_B[i][tx];
        }

        a += a_step;
        b += b_step;
    }

    // check if out of bound
    if (by * BLOCK_SIZE + ty >= height_A) return;
    if (bx * BLOCK_SIZE + tx >= width_B) return;

    // set result in global memory
    int c_index = \
        by * width_B * BLOCK_SIZE + \
        bx * BLOCK_SIZE + \
        ty * width_B + \
        tx;
    C[c_index] = sum;
}

int main(int argc, char** argv) {
    // matrix size
    int height_A, width_A;
    int height_B, width_B;

    printf("Matrix A height: "); scanf("%d", &height_A);
    printf("Matrix A width: "); scanf("%d", &width_A);
    printf("Matrix B height: "); scanf("%d", &height_B);
    printf("Matrix B width: "); scanf("%d", &width_B);

    if (height_A <= 0 || width_A <= 0 ||
        height_B <= 0 || width_B <= 0) {
        printf("Invalid matrix size\n");
        return 1;
    } else if (width_A != height_B) {
        printf("width_A != height_B\n");
        return 1;
    }

    // result size
    int height_C = height_A;
    int width_C = width_B;

    // alloc host
    int size_A = height_A * width_A;
    int size_B = height_B * width_B;
    int size_C = height_C * width_C;

    int mem_A = sizeof(float) * size_A;
    int mem_B = sizeof(float) * size_B;
    int mem_C = sizeof(float) * size_C;

    float *host_A = new float[size_A];
    float *host_B = new float[size_B];
    float *host_C = new float[size_C];

    // init host
#ifdef DEBUG
    srand(0);
#else
    srand(time(NULL));
#endif

    for (int i = 0; i < size_A; ++i)
        host_A[i] = rand() / (float)RAND_MAX;

    for (int i = 0; i < size_B; ++i)
        host_B[i] = rand() / (float)RAND_MAX;

    for (int i = 0; i < size_C; ++i)
        host_C[i] = 0.0;

    // alloc device
    float *dev_A;
    float *dev_B;
    float *dev_C;
    cudaMalloc(&dev_A, mem_A);
    cudaMalloc(&dev_B, mem_B);
    cudaMalloc(&dev_C, mem_C);

    // copy memory
    cudaMemcpy(dev_A, host_A, mem_A, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, host_B, mem_B, cudaMemcpyHostToDevice);

    // perform CUDA
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (width_C / block.x) + (width_C % block.x != 0),
        (height_C / block.y) + (height_C % block.y != 0));

    struct timeval tic, toc;
    gettimeofday(&tic, NULL);
    matmul<<<grid,block>>>(
        dev_A, dev_B, dev_C,
        height_A, width_A,
        height_B, width_B);
    cudaDeviceSynchronize();
    gettimeofday(&toc, NULL);

    double elapsed = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
    printf("(CUDA) [%dx%d]*[%dx%d]: %lfs (%lf GFlops)\n",
        height_A, width_A,
        height_B, width_B,
        elapsed,
        2.*height_A*width_A*width_B/elapsed/1e9);

    // copy result
    cudaMemcpy(host_C, dev_C, mem_C, cudaMemcpyDeviceToHost);

    // perform CPU
    float *host_D = new float[size_C];

    gettimeofday(&tic, NULL);
#pragma omp parallel for
    for (int i = 0; i < height_A; ++i) {
        for (int j = 0; j < width_B; ++j) {
        #ifdef DEBUG
            float sum = 0.0;
            for (int k = 0; k < height_B; ++k) {
                sum += host_A[i*width_A + k] * host_B[k*width_B + j];
            }
            host_D[width_B*i + j] = sum;
        #else
            for (int k = 0; k < height_B; ++k) {
                host_C[width_B*i + j] -= host_A[i*width_A + k] * host_B[k*width_B + j];
            }
        #endif
        }
    }
    gettimeofday(&toc, NULL);

    elapsed = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
    printf("(CPU) [%dx%d]*[%dx%d]: %lfs (%lf GFlops)\n",
        height_A, width_A,
        height_B, width_B,
        elapsed,
        2.*height_A*width_A*width_B/elapsed/1e9);

    // calc error
    float err = 0.0;
    for (int i = 0; i < size_C; ++i)
    #ifdef DEBUG
        err += fabs(host_C[i]-host_D[i]);
    #else
        err += fabs(host_C[i]);
    #endif
    printf("error: %f\n",err/size_C);

#ifdef DEBUG
    printf("\nDEBUG");
    printf("\n\nA:\n"); for (int i = 0; i < size_A; ++i) printf("%f ", host_A[i]);
    printf("\n\nB:\n"); for (int i = 0; i < size_B; ++i) printf("%f ", host_B[i]);
    printf("\n\nC:\n"); for (int i = 0; i < size_C; ++i) printf("%f ", host_C[i]);
    printf("\n\nD:\n"); for (int i = 0; i < size_C; ++i) printf("%f ", host_D[i]);
#endif

    // free
    delete[] host_A;
    delete[] host_B;
    delete[] host_C;
    delete[] host_D;
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    return 0;
}

