include $(GOROOT)/src/Make.inc

template: *.go
	$(GC) -I_obj *.go 
	$(LD) -L_obj -o $@ $@.$O


.PHONY:	clean
clean:
	rm -f  template *.a *.$O $(CLEANFILES)

.PHONY: install
install:

.PHONY: test
test: 

.PHONY: bench
bench:

