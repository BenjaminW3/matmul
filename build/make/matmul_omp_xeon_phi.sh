$ export MIC_ENV_PREFIX=MIC
$ export MIC_OMP_NUM_THREADS=60
$ export MIC_KMP_AFFINITY=compact, granularity=fine, verbose  # compact/scatter/balanced/verbose
$ export MKL_MIC_ENABLE=1


./matmul
