FLAGS = -MMD -MP -Iinclude -lm -g

SRC = $(wildcard test/*.c)
BIN = $(patsubst test/%.c,bin/%,$(SRC))

all: $(BIN)

bin/%: test/%.c | bin
	@echo "Building $< ..."
	@gcc $< $(FLAGS) -o $@

bin:
	@echo "Creating bin directory ..."
	@mkdir -p bin

run: $(BIN)
	@for b in $(BIN); do \
		echo "==== Running $$b ===="; \
		./$$b; \
		echo; \
	done

pipeline: clean run

clean:
	@echo "cleaning bin ..."
	@rm -rf bin

-include $(BIN:=.d)
