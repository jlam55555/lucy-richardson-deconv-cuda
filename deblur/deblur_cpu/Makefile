BINARY:=deblur
SOURCES:=$(shell find *.c)
OBJECTS:=$(SOURCES:%.c=%.o)

CFLAGS:=$(shell pkg-config --cflags libpng)
LDFLAGS:=$(shell pkg-config --libs libpng) -lm

$(BINARY): $(OBJECTS)
	$(CC) -o $(BINARY) $(OBJECTS) $(LDFLAGS) 

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

# target-specific variable values: https://stackoverflow.com/a/1080180
debug: CFLAGS+= -g
debug: $(BINARY)

.PHONY:
clean:
	rm -f $(BINARY) *.o
