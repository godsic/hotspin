include src/Make.inc

export GOPATH=$(CURDIR)/

LIBNAME=libhotspin.so
export CUDAROOT=/usr/local/cuda
export NVROOT=/usr/lib64/nvidia-bumblebee
export CUDA_INC_PATH=$(CUDAROOT)/include/
export CUDA_LIB_PATH=$(NVROOT):$(CUDAROOT)/lib64/

all:
	@echo $(GOPATH)
	$(MAKE) --no-print-directory --directory=src/libhotspin 
	cp src/libhotspin/$(LIBNAME) src/hotspin-core/gpu/
	cp src/libhotspin/$(LIBNAME) .
	cp src/libhotspin/$(LIBNAME) bin/
	go run src/cuda/setup-cuda-paths.go -dir=src/cuda/
	go install -race -compiler=gccgo -gccgoflags='-static-libgcc -L /home/mykola/systools/hotspins/pkg/gccgo -L /usr/local/cuda/lib64/ -O3 -march=native' -v hotspin
	#go install -v hotspin
	go install -v apigen
	go install -v texgen
	go install -v template
	make -C src/python
	cp -rf src/python/* ./python/

.PHONY: clean
clean:	
	rm -rf python/*
	rm -rf pkg/*
	rm -rf src/hotspin-core/gpu/$(LIBNAME)
	rm $(LIBNAME)
	rm -rf bin/hotspin
	rm -rf bin/apigen
	rm -rf bin/texgen
	rm -rf bin/$(LIBNAME)
	make clean -C src/python
	make clean -C src/libhotspin
.PHONY: test
test:
	echo todo
		
.PHONY: tidy	
tidy:
	@find * | egrep "#" | xargs rm -f
	@find * | egrep "\~" | xargs rm -f

.PHONY: love	
love:
	@echo Oh, yeah
	@echo Do it again to hotspin!
	
.PHONY: doc
doc:

	make -C doc
	ln -sf doc/manual/manual.pdf hotspin-manual.pdf

.PHONY: githooks
githooks:
	ln -sf $(CURDIR)/misc/pre-commit .git/hooks 


