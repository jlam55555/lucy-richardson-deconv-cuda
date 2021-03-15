# simple generic makefile for compiling all .cu files in the
# current directory

CC:=/usr/local/cuda/bin/nvcc
BINARY:=blur

BUILDDIR:=build

SOURCES:=$(shell ls *.cu)
OBJECTS:=$(addprefix $(BUILDDIR)/,$(SOURCES:.cu=.o))

$(BINARY): $(BUILDDIR) $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS)

.PHONY:
$(BUILDDIR):
	mkdir -p $(BUILDDIR)
	echo $(SOURCES)
	echo $(OBJECTS)

$(BUILDDIR)/%.o: %.cu
	$(CC) -c -o $@ $<

.PHONY:
clean:
	rm -rf $(BUILDDIR) $(BINARY)
