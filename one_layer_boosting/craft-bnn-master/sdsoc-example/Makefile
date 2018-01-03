CXX=g++
CFLAGS=-O3 -std=gnu++11 -g

# HDR are pure headers
HDR=
# OBJ must include a .cpp and .h with same name
OBJ=Top.o
EXE=test.exe

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
	vivado_hls hls.tcl

hlsclean:
	rm -rf hls.prj vivado_hls.log

clean: hlsclean
	rm -f *.o *.exe
