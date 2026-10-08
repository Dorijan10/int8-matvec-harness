CC = gcc
# No -O3 or -march=native, so an optimiser has headroom over the naive kernel.
CFLAGS = -O2 -std=c11 -Wall -Wextra
BIN = build/matvec
SRC = src/main.c src/matvec.c src/reference.c

.PHONY: compile test bench clean

compile: $(BIN)

$(BIN): $(SRC)
	mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC)

test: $(BIN)
	./$(BIN) test

bench: $(BIN)
	./$(BIN) bench

clean:
	rm -rf build artemis_results.json
