# build all subprograms

.PHONY:
all:
	$(MAKE) -C deblur_cpu
	$(MAKE) -C deblur_cuda
	$(MAKE) -C deblur_eval

.PHONY:
clean:
	$(MAKE) -C deblur_cpu clean
	$(MAKE) -C deblur_cuda clean
	$(MAKE) -C deblur_eval clean
