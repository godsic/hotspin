export GOPATH=$(CURDIR)/../..

DIRS=driver runtime cufft curand 

all: $(DIRS)

.PHONY: $(DIRS)
$(DIRS): 
		go run setup-cuda-paths.go
		go install -v cuda/$@

