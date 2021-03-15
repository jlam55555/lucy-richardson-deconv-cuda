Sample vectorAdd application for use with Jetson Nano

Can be run on advcomparch.ee.cooper.edu (if not available: comparch.ee.cooper.edu - advcomparch will be displayed as hostname when logged in)

Compilation steps:
- make vectorAdd
- No environment variables need to be set to either compile on Jetson or cross compile for Jetson
- To compile for x86 with different GPU, set TARGET_ARCH to x86_64 and SMS to the appropriate value for the desired GPU (see https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ for examples)
- A vectorAdd binary will be produced which can run on the Jetson Nano
- By default, files are saved in user's home directory and will be immediately available on the Jetson

To build alternate applications, create appropriate makefile rules copying the example flags for either binary or object generation.