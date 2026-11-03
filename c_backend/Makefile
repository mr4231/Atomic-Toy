CC ?= gcc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic
LDFLAGS ?= -lm

.PHONY: test optimized debug clean

test: pairforce.c test_pairforce.c pairforce.h
	$(CC) $(CFLAGS) -O2 pairforce.c test_pairforce.c -o test_pairforce $(LDFLAGS)
	./test_pairforce

optimized: pairforce.c pairforce.h
	$(CC) $(CFLAGS) -O2 -shared -fPIC pairforce.c -o pairforce.so $(LDFLAGS)

debug: pairforce.c pairforce.h
	$(CC) $(CFLAGS) -O0 -g -shared -fPIC pairforce.c -o pairforce_debug.so $(LDFLAGS)

clean:
	rm -f test_pairforce pairforce.so pairforce_debug.so pairforce.dll pairforce_debug.dll
