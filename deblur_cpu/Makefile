BINARY:=deblur
SOURCES:=$(shell find *.c)
OBJECTS:=$(SOURCES:%.c=%.o)

CFLAGS:=$(shell pkg-config --cflags libpng)
LDFLAGS:=$(shell pkg-config --libs libpng)

$(BINARY): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $(BINARY) $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

.PHONY:
clean:
	rm -f $(BINARY) *.o
