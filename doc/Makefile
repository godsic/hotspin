.SILENT:

godirs=\
	hotspin-core\
	hotspin-core/common\
	hotspin-core/gpu\
	cuda\
	cuda/driver\

all: doxygen manual
# $(godirs)

doxygen:
	doxygen Doxyfile||echo "make doc requires package doxygen, graphviz"

.PHONY: manual
manual:
	make -C manual

.PHONY: $(godirs)
$(godirs):
	mkdir -p html/$@
	#ln -sf $(shell readlink -f ../src/pkg/)/$@ $(GOROOT)/src/pkg/$@
	godoc -path=. -html $@ > html/$@/index.html

cu:

.PHONY: clean
clean:
	rm -rf html latex
	make clean -C manual

.PHONY: install
test:

.PHONY: test
test:

.PHONY: bench
test:

