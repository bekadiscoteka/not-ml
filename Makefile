bin/out: nomain.c | bin
	@echo "Building the file ..."
	@gcc $^ -Iinclude -o $@

bin:
	@echo "Creating bin directory ..."
	@mkdir -p bin

pipeline: clean bin/out run

clean:
	@echo "cleaning bin ..."
	@rm -rf bin

run:
	@echo "Running!"
	@bin/out



