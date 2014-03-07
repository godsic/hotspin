include $(GOROOT)/src/Make.inc

executables=\
	mumax2\
	apigen\
	texgen\

all: $(executables) 

modules=$(wildcard ../*.mod)

# always build executables
# build is fast and dependencies complex
.PHONY: $(executables)

mumax2 texgen:
	$(GC) -I_obj $@.go $(modules:.mod=.mod/pkg/import.go)
	$(LD) -L_obj -o $@ $@.$O

apigen:
	$(GC) -I_obj $@.go
	$(LD) -L_obj -o $@ $@.$O

.PHONY:	clean
clean:
	rm -f  $(executables) *.a *.$O $(CLEANFILES)

.PHONY: install
install:

.PHONY: test
test: 

.PHONY: bench
bench:

