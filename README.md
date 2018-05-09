# High Performance Scientific Computing - Matrix Multiplication

- Name        : Tifani Warnita
- Student ID  : 17M38271

## 1. Execution Instruction
### A. BLISLab Version
1. Go to login node of TSUBAME. 
2. Move to folder `src/blislab`.
3. Type, `source sourceme.h`
4. Load Intel module, `module load intel`.
5. Build the files, `make`. We will have two executable files:
    - Baseline from the class: `dgemm_baseline.x`
    - Optimized code: `dgemm_optimized.x`
6. Execute the program:
    - `./dgemm_baseline.x [M] [N] [K]`
    - `./dgemm_optimized.x [M] [N] [K]`

### B. CUDA Version
1. Go to q_node of TSUBAME, `qrsh -g tga-hpc-lecture -l q_node=1 -l h_rt=0:10:00`.
2. Move to folder `src/cuda`.
3. Load CUDA module, `module load cuda`.
4. Build the files, `make`. We will have two executable files:
    - Baseline from the class: `./cuda_baseline.x [M]`
    - Optimized code: `./cuda_optimized.x`
   
   
## 2. Result Comparison (best size so far) 
### A. BLISLAB Version
- `./dgemm_baseline.x 5000 5000 5000`
    5000	  5000	  5000	 240.78	 404.06
- `./dgemm_optimized.x 5000 5000 5000`
    5000	  5000	  5000	 433.63	 409.06

### B. CUDA Version
- `./cuda_baseline.x 1024`
 
  N=1024: 0.003412 s (629.391456 GFlops)
  
  N=1024: 0.109662 s (19.582751 GFlops)
  
  error: 0.000129
  
- `./cuda_optimized.x`
  
  (CUDA) [1024x1024]*[1024x1024]: 0.001048s (2049.125618 GFlops)
  
  (CPU) [1024x1024]*[1024x1024]: 0.421494s (5.094933 GFlops)
  
  error: 0.000129
