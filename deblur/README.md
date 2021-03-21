# deblur

### Build instructions
Use the Makefile in this directory to build everything.
```bash
$ make [all] [clean]
```

You will still need to pass in any system-specific CUDA build parameters, e.g.:
```
$ make all CUDA_PATH=/usr TARGET_ARCH=x86_64 SMS=30
```

Or, to build it each program manually:

##### CPU-only programs
```bash
$ make -C deblur_cpu [TARGET]
$ make -C deblur_eval [TARGET]
```

##### CUDA application
For the CUDA application, you may need to specify the appropriate
flags for the Makefile (see an example in [buildx86.sh][buildx86])
```bash
$ make -C deblur_cuda CUDA_PATH=/usr TARGET_ARCH=x86_64 SMS=30 [TARGET]
```

### Report
coming soon...

### Results

##### Performance
Tests performed on [earth_blurry.png][infile] 25 rounds RL w/ gaussian kernel (blur std=3)

CPU version, i7-2600, 
```bash
$ deblur_cpu/deblur samples/earth_blurry.png samples/earth_deblurred_cpu_std3.png 25 3
overall: 219.035156s
round: 8.761112s
conv2d: 4.369866
mult/div: 0.010680
```

CUDA version, GT740
```bash
$ deblur_cuda/deblur samples/earth_blurry.png samples/earth_deblurred_cuda_std3.png 25 3
overall: 5.92944s
round: 0.237075s
conv2d: 0.11415s
mult/div: 0.0043831s
```

##### Attempting different blur radii (CUDA only)
Gaussian kernel std
```bash
$ deblur_cuda/deblur samples/earth_blurry.png samples/earth_deblurred_cuda_std1.png 25 1
overall: 1.09431s
round: 0.043669s
conv2d: 0.0174834s
mult/div: 0.00434682s
$ deblur_cuda/deblur samples/earth_blurry.png samples/earth_deblurred_cuda_std2.png 25 2
overall: 3.00892s
round: 0.120253s
conv2d: 0.0557258s
mult/div: 0.00439642s
$ deblur_cuda/deblur samples/earth_blurry.png samples/earth_deblurred_cuda_std4.png 25 4
overall: 9.9609s
round: 0.398333s
conv2d: 0.194796s
mult/div: 0.00436604s
```

##### Accuracy
Between blurred and deblurred versions
```bash
$ deblur_eval/eval_error samples/earth.png samples/*.png
MSE samples/earth.png <-> samples/earth_blurry.png (per datapoint): 1247.220825
MSE samples/earth.png <-> samples/earth_deblurred_cpu_std3.png (per datapoint): 1380.085327
MSE samples/earth.png <-> samples/earth_deblurred_cuda_std1.png (per datapoint): 1264.200317
MSE samples/earth.png <-> samples/earth_deblurred_cuda_std2.png (per datapoint): 1273.957153
MSE samples/earth.png <-> samples/earth_deblurred_cuda_std3.png (per datapoint): 1380.085693
MSE samples/earth.png <-> samples/earth_deblurred_cuda_std4.png (per datapoint): 1537.670410
MSE samples/earth.png <-> samples/earth.png (per datapoint): 0.000000
```

Between CPU and CUDA version
```bash
$ deblur_eval/eval_error samples/earth_deblurred_*_std3.png
MSE samples/earth_deblurred_cpu_std3.png <-> samples/earth_deblurred_cuda_std3.png (per datapoint): 0.000014
```

##### Deblur effectiveness
```bash
$ deblur_eval/eval_sharpness samples/*.png
Sharpness measure of samples/earth_blurry.png: 8.361764
Sharpness measure of samples/earth_deblurred_cpu_std3.png: 53.256336
Sharpness measure of samples/earth_deblurred_cuda_std1.png: 22.957777
Sharpness measure of samples/earth_deblurred_cuda_std2.png: 36.938744
Sharpness measure of samples/earth_deblurred_cuda_std3.png: 53.256336
Sharpness measure of samples/earth_deblurred_cuda_std4.png: 50.872707
Sharpness measure of samples/earth.png: 1694.936890
```

[infile]: samples/earth_blurry.png
[buildx86]: deblur_cuda/buildx86.sh
