#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>

#define GEMM_SIMD_ALIGN_SIZE 32
#define SGEMM_MC 144
#define SGEMM_NC 6800
#define SGEMM_KC 512
#define SGEMM_MR 24
#define SGEMM_NR 4

#define y0 -1
#define y1 1
#define theta 0
#define eps 1

typedef unsigned long long dim_t;

struct aux_t {
  float *b_next;
  float *b_next_s;
  char  *flag;
  int   pc;
  int   m;
  int   n;
};

extern "C" void sgemm_(char*, char*, int*, int*, int*, float*, float*,
                       int*, float*, int*, float*, float*, int*);


float *bl_malloc_aligned(int m, int n, int size) {
  float *ptr;
  int err = posix_memalign((void**)&ptr,(size_t)GEMM_SIMD_ALIGN_SIZE, size*m*n);
  if (err) {
    printf( "bl_malloc_aligned(): posix_memalign() failures" );
    exit(1);
  }
  return ptr;
}

inline void packA_mcxkc_d(int m, int k, float *XA, int ldXA, int offseta, float *packA) {
  float *a_pntr[SGEMM_MR];
  for (int i=0; i<m; i++) {
    a_pntr[i] = XA + offseta + i;
  }
  for (int i=m; i<SGEMM_MR; i++) {
    a_pntr[i] = XA + offseta + 0;
  }
  for (int p=0; p<k; p++) {
    for (int i=0; i<SGEMM_MR; i++) {
      *packA = *a_pntr[i];
      packA++;
      a_pntr[i] = a_pntr[i] + ldXA;
    }
  }
}

inline void packB_kcxnc_d(int n, int k, float *XB, int ldXB, int offsetb, float *packB) {
  float *b_pntr[SGEMM_NR];
  for (int j=0; j<n; j++) {
    b_pntr[j] = XB + ldXB * (offsetb + j);
  }
  for (int j=n; j<SGEMM_NR; j++) {
    b_pntr[j] = XB + ldXB * (offsetb + 0);
  }

  for (int p=0; p<k; p++) {
    for (int j=0; j<SGEMM_NR; j++) {
      *packB++ = *b_pntr[j]++;
    }
  }
}

#if 1
#include "micro_kernel.h"
#else
void micro_kernel(int kc, float* A, float* B, float* C, dim_t ldc, aux_t* aux) {
  int nr = aux->n;
  int mr = aux->m;
  for (int kr=0; kr<kc; kr++) {
    for (int j=0; j<nr; j++) {
      for (int i=0; i<mr; i++) {
        C[j*ldc+i] += A[i*kc+kr] * B[j*kc+kr];
      }
    }
  }
}
#endif

void macro_kernel(int mc, int nc, int kc, float *packA, float *packB, float *C, int ldc) {
  aux_t aux;
  aux.b_next = packB;

  int bl_ic_nt = 16;
  char *str = getenv( "BLISLAB_IC_NT" );
  if ( str != NULL ) {
    bl_ic_nt = (int)strtol(str, NULL, 10);
  }

#pragma omp parallel for num_threads(bl_ic_nt)
  for (int jr=0; jr<nc; jr+=SGEMM_NR) {                 // 2-th loop around micro-kernel (NR)
    aux.n = std::min(nc-jr, SGEMM_NR);
    for (int ir=0; ir<mc; ir+=SGEMM_MR) {               // 1-st loop around micro-kernel (MR)
      aux.m = std::min(mc-ir, SGEMM_MR);
      if (ir+SGEMM_MR >= mc) {
        aux.b_next += SGEMM_NR * kc;
      }
      micro_kernel(kc, &packA[ir*kc], &packB[jr*kc], &C[jr*ldc+ir], ldc, &aux);
    }                                                   // 2-nd loop around micro-kernel
  }                                                     // 1-st loop around micro-kernel
}

void bl_sgemm(int m, int n, int k, float *XA, int lda, float *XB, int ldb, float *C, int ldc) {
  int bl_ic_nt = 16;
  char *str = getenv( "BLISLAB_IC_NT" );
  if ( str != NULL ) {
    bl_ic_nt = (int)strtol(str, NULL, 10);
  }
  float * packA = bl_malloc_aligned(SGEMM_KC, (SGEMM_MC+1)*bl_ic_nt, sizeof(float));
  float * packB = bl_malloc_aligned(SGEMM_KC, (SGEMM_NC+1)         , sizeof(float));
  for (int jc=0; jc<n; jc+=SGEMM_NC) {                  // 5-th loop around micro-kernel (NC)
    int nc = std::min(n-jc, SGEMM_NC);                    // [jc..jc+NC-1]
    for (int pc=0; pc<k; pc+=SGEMM_KC) {                // 4-th loop around micro-kernel (KC)
      int kc = std::min(k-pc, SGEMM_KC);                  // [pc..pc+KC-1]
#pragma omp parallel for num_threads(bl_ic_nt)
      for (int j=0; j<nc; j+=SGEMM_NR) {
        packB_kcxnc_d(std::min(nc-j, SGEMM_NR), kc, &XB[pc], k, jc+j, &packB[j*kc]);  // size: KC x NC (why ldXB = k)
      }
#pragma omp parallel for num_threads(bl_ic_nt)
      for (int ic=0; ic<m; ic+=SGEMM_MC) {              // 3-rd loop around micro-kernel (MC)
        int tid = omp_get_thread_num();
        int mc = std::min(m-ic, SGEMM_MC);
        for (int i=0; i<mc; i+=SGEMM_MR) {
          packA_mcxkc_d(std::min(mc-i, SGEMM_MR), kc, &XA[pc*lda], m, ic+i, &packA[tid*SGEMM_MC*kc+i*kc]);
        }
        macro_kernel(mc, nc, kc, packA+tid*SGEMM_MC*kc, packB, &C[jc*ldc+ic], ldc);
      }                                                 // end of 3-rd loop around micro-kernel (MC)
    }                                                   // end of 4-th loop around micro-kernel (KC)        
  }                                                     // end of 5-th loop around micro-kernel (NC)
  free(packA);
  free(packB);
}

#define A( i, j ) A[(j)*lda+(i)]
#define B( i, j ) B[(j)*ldb+(i)]
#define C( i, j ) C[(j)*ldc+(i)]
#define C_ref( i, j ) C_ref[(j)*ldc_ref+(i)]

int main(int argc, char *argv[]) {
  int m, n, k;
  sscanf(argv[1], "%d", &m);
  sscanf(argv[2], "%d", &n);
  sscanf(argv[3], "%d", &k);
  float alpha = 1.0, beta = 1.0;
  double ref_rectime, bl_sgemm_rectime;
  float * A = (float*)malloc(sizeof(float)*m*k);
  float * B = (float*)malloc(sizeof(float)*k*n);
  int lda = m;
  int ldb = k;
  int ldc = ((m-1) / SGEMM_MR + 1) * SGEMM_MR;
  int ldc_ref = m;
  float * C = bl_malloc_aligned(ldc, n+4, sizeof(float));
  float * C_ref = (float*)malloc(sizeof(float)*m*n);
  int nrepeats = 3;
  srand(time(NULL));
  for (int p=0; p<k; p++) {
    for (int i=0; i<m; i++) {
      A(i,p) = drand48();
    }
  }
  for (int j=0; j<n; j++) {
    for (int p=0; p<k; p++) {
      B(p,j) = drand48();
    }
  }
  for (int j=0; j<n; j++) {
    for (int i=0; i<m; i++) {
      C_ref(i,j) = 0.0;
      C(i,j) = 0.0;
    }
  }
  for (int i=0; i<nrepeats; i++) {
    double bl_sgemm_beg = omp_get_wtime();
    bl_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
    double bl_sgemm_time = omp_get_wtime() - bl_sgemm_beg;
    if (i == 0) {
      bl_sgemm_rectime = bl_sgemm_time;
    } else {
      bl_sgemm_rectime = bl_sgemm_time < bl_sgemm_rectime ? bl_sgemm_time : bl_sgemm_rectime;
    }
  }
  for (int i=0; i<nrepeats; i++) {
    double ref_beg = omp_get_wtime();
    sgemm_( "N", "N", &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C_ref, &ldc_ref );
    double ref_time = omp_get_wtime() - ref_beg;
    if (i == 0) {
      ref_rectime = ref_time;
    } else {
      ref_rectime = ref_time < ref_rectime ? ref_time : ref_rectime;
    }
  }
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      if (fabs(C(i,j) - C_ref(i,j)) > 1) {
        printf( "C[ %d ][ %d ] != C_ref, %E, %E\n", i, j, C(i,j), C_ref(i,j));
        break;
      }
    }
  }
  float flops = (m * n / (1000.0 * 1000.0 * 1000.0)) * (2 * k);
  printf("%5d\t %5d\t %5d\t %5.2lf\t %5.2lf\n",
         m, n, k, flops / bl_sgemm_rectime, flops / ref_rectime);
  free(A);
  free(B);
  free(C);
  free(C_ref);
  return 0;
}

