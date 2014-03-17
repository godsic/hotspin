export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../libhotspin

APIGEN=./apigen.sh

all: hotspin.py

.PHONY: hotspin.py
hotspin.py: hotspin.py.tail 
	$(APIGEN) $(CURDIR)/../hotspin-core/engine/api.go
	mv hotspin.py hotspin.py.head
	cat hotspin.py.head hotspin.py.tail > hotspin.py
	rm -f hotspin.py.head

CLEANFILES=\
	hotspin.py\
	hotspin.pyc\
	hotspin.lua\
	Hotspin.java\
	Hotspin.class\


.PHONY: clean
clean:
	rm -rf $(CLEANFILES)

.PHONY: install
install: all

.PHONY: test
test: 

.PHONY: bench
bench:

