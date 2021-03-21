# simple makefile to compile single files into their binaries with the
# same name
CFLAGS:=$(shell pkg-config --cflags libpng)
LDFLAGS:=$(shell pkg-config --libs libpng) -lm

# binaries to build
BINARIES:=eval_error eval_sharpness

.PHONY:
all: $(BINARIES)

# static pattern rule: https://stackoverflow.com/a/15718720
$(BINARIES): %: %.o pngio.o
	$(CC) -o $@ $< pngio.o $(LDFLAGS) 

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

# target-specific variable values: https://stackoverflow.com/a/1080180
debug: CFLAGS+= -g
debug: $(BINARIES)

.PHONY:
clean:
	rm -f $(BINARIES) *.o
