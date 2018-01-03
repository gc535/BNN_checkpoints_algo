include ../Makefile.inc

CFLAGS:=-I../minizip -I../utils $(CFLAGS)
LDFLAGS:=-L../utils -lCraftUtils $(LDFLAGS)

# HDR are pure headers
HDR=
# OBJ must include a .cpp and .h with same name
OBJ=Cifar10.o
OLD_EXE=quantize_params_kh.exe quantize_params_nb.exe
EXE=cifar10_test_layer.exe cifar10_test_accuracy.exe

all: $(EXE)

# Rule for object files, each object must have a header
$(OBJ): %.o: %.cpp %.h
	$(CXX) -c $< -o $@ $(CFLAGS)

%.o: %.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

# Rule for executables
$(EXE): %.exe: %.o $(OBJ)
	g++ $^ -o $@ $(CFLAGS) $(LDFLAGS)

.PHONY: hls clean hlsclean
hls:
	#vivado_hls hls.tcl
	@echo "HLS of the C++ baseline takes over 24 hours, disabled by default"

hlsclean:
	rm -rf hls.prj vivado_hls.log

clean: hlsclean
	rm -f *.o *.exe
