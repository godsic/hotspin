export GOPATH=$(CURDIR)/../..

DIRS=apigen common engine frontend gpu host modules ovf queue dump

all: $(DIRS)

.PHONY: $(DIRS)
$(DIRS): 
		go install -v mumax/$@
