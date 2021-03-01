# simple generic makefile for compiling all .cu files in the
# current directory

CC:=/usr/local/cuda/bin/nvcc
BINARY:=blur
BUILDDIR:=build

LIBS:=libjpeg

SOURCES:=$(shell ls *.cu)
OBJECTS:=$(addprefix $(BUILDDIR)/,$(SOURCES:.cu=.o))
LIBFLAGS:=$(shell pkg-config --libs $(LIBS))
COMPILEFLAGS:=$(shell pkg-config --cflags $(LIBS)) -g

$(BINARY): $(BUILDDIR) $(OBJECTS)
	$(CC) $(LIBFLAGS) -o $(BINARY) $(OBJECTS)

.PHONY:
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: %.cu
	$(CC) $(COMPILEFLAGS) -c -o $@ $<

.PHONY:
clean:
	rm -rf $(BUILDDIR) $(BINARY)
