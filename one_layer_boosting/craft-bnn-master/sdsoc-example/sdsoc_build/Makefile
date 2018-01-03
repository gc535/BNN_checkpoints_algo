SDSFLAGS=-sds-pf zed -sds-hw top Top.cpp -hls-tcl sds.tcl -sds-end
CXX=sds++ $(SDSFLAGS)
CFLAGS=-O3 -std=gnu++11 -g

# HDR are pure headers
HDR=
# OBJ must include a .cpp and .h with same name
OBJ=Top.o
EXE=test.exe

all: $(EXE)

# Rule for object files, each object must have a header
$(OBJ): %.o: ../%.cpp ../%.h
	$(CXX) -c $< -o $@ $(CFLAGS)

%.o: ../%.cpp
	$(CXX) -c $< -o $@ $(CFLAGS)

# Rule for executables
$(EXE): %.exe: %.o $(OBJ)
	$(CXX) $^ -o $@ $(CFLAGS) $(LDFLAGS)

.PHONY: hls clean hlsclean
clean:
	rm -f *.o *.exe
	rm -rf _sds sd_card
