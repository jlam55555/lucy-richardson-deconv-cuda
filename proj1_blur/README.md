# Project 1: Blurring
and learning how to use CUDA

### Instructions
##### Compile
```bash
$ make
```
See the [Makefile](./Makefile).

##### Run
```bash
$ ./blur res/sample.jpg sample_blurred.jpg 100
```
The first parameter is the input file, the second parameter is the output
file, and the final parameter is the blur radius.

### Image Support
For now, only JPEG images are supported. [libjpeg][libjpeg] is used for JPEG
compression.

(In hindsight, it would be much easier to use a multitalented
library like ImageMagick's C API to handle many more images, but it was fun
to play around with lower-level libraries like libjpeg.)

(TODO: add PNG support)

### Report
(TODO: the report is currently nonexistent)

[libjpeg]: http://libjpeg.sourceforge.net/
