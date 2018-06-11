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
    - Optimized code:
        - Square matrix: `./cuda_optimized.x [M]`
        - Any matrix   : `./cuda_optimized.x [M] [N] [K]`
   
   
## 2. Results

|       Category      |        Size        |  Gflops |
|:-------------------:|:------------------:|:-------:|
| BLISLAB - Baseline  | 5000 x 5000 x 5000 |  240.78 |
| BLISLAB - Optimized | 5000 x 5000 x 5000 |  433.63 |
| CUDA - Baseline     | 4096 x 4096 x 4096 |  288.16 |
| CUDA - Optimized    | 4096 x 4096 x 4096 | 2305.91 |

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
  
- `./cuda_optimized.x 1024`  
    [1024x1024x1024]  
    CUDA  : 0.001046s (2053.043641 GFlops)  
    CPU   : 0.388411s (5.528895 GFlops)  
    Error : 0.010121
    
- `./cuda_baseline.x 4096`  
    N=4096: 0.476958 s (288.157350 GFlops)  
    N=4096: 4.614974 s (29.781089 GFlops)  
    error: 0.001008
    
- `./cuda_optimized.x 4096`  
    [4096x4096x4096]  
    CUDA  : 0.059603s (2305.906640 GFlops)  
    CPU   : 91.365267s (1.504280 GFlops)  
    Error : 0.007812  
