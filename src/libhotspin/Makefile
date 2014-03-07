include ../Make.inc

CUFILES   = $(filter-out $(wildcard $(MY_CUFILES_WERRORLESS)), $(wildcard *.cu))  # my cuda files
CPPFILES     = $(wildcard *.cpp)                 # my c++ files
CU_HFILES    = $(CUFILES:.cu=.h)                 # cuda only header files
CU_HFILES   += $(MY_CUFILES_WERRORLESS:.cu=.h)                 # cuda only header files
HFILES       = $(CUHFILES) $(CPPFILES:.cpp=.h)   # all header files

ifndef SystemRoot
CU_OFILES    = $(CUFILES:.cu=.o)              	# my cuda objects
CPP_OFILES   = $(CPPFILES:.cpp=.o)              # c++ objects
else
CU_OFILES    = $(CUFILES:.cu=.obj)              
CPP_OFILES   = $(CPPFILES:.cpp=.obj)  
endif

OFILES       = $(CU_OFILES) $(CPP_OFILES)\
			   $(CU_OFILES_WERRORLESS)           # all objects


ifndef SystemRoot

all: libhotspin.so libhotspin.h

$(CU_OFILES): %.o: %.cu %.h
	$(NVCC) -dc -I. $(@:.o=.cu) -o $(@)

$(CPP_OFILES): %.o: %.cpp %.h
	 $(CC) -fPIC -c $(@:.o=.cpp) -o $(@)

libhotspin.so: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) -shared $(OFILES) -o libhotspin.so

libhotspin.h: $(CU_HFILES)
	cp libhotspin.h.head libhotspin.h
	$(foreach hfile, $(CU_HFILES),  echo \#include "\""$(hfile)"\"" >> libhotspin.h;)
	cat libhotspin.h.tail >> libhotspin.h	
	
else

all: libhotspin.lib libhotspin.h

$(CU_OFILES): %.obj: %.cu %.h
	$(NVCC) --compiler-options -D_USRDLL --compiler-options -DDLL_EXPORTS -dc -I. $(@:.obj=.cu) -o $(@)

$(CPP_OFILES): %.obj: %.cpp %.h
	 $(CC) /D_USRDLL /DDLL_EXPORTS -c $(@:.obj=.cpp) -o $(@)

libhotspin.lib: $(OFILES) 
	echo " *** USING DEBUG FLAGS *** "
	$(NVCC) --shared --linker-options -DLL $(OFILES) -o libhotspin.dll
	cp libhotspin.dll ../../bin/
	
libhotspin.h: $(CU_HFILES)
	cp libhotspin.h.head libhotspin.h.bak
	$(foreach hfile, $(CU_HFILES), echo #include "$(hfile)" >> libhotspin.h.bak)
	cat libhotspin.h.tail >> libhotspin.h.bak
	cat libhotspin.h.bak | sed "s/echo /\n/g" >> libhotspin.h
	
endif
	
.PHONY: link
link:
	ln -sf $(CURDIR)/libhotspin.so ../pkg/hotspin-core/gpu
	$(foreach module, $(wildcard ../*.mod), ln -sf $(CURDIR)/libhotspin.so $(module)/pkg;)

.PHONY: clean
clean:
ifndef SystemRoot
	rm -f *.o *.so libhotspin.h ../*.mod/cuda/*.o
else
	rm -f *.def *.exp *.obj *.lib *.dll *.pdb libhotspin.h 
	rm -f -R ../*.mod/cuda/*.obj
endif

install: all

.PHONY: test
test:

.PHONY: bench
bench:

