main.exe: main.cpp
	g++ -g --std=gnu++11 -O1 $< -o $@
hls:
	vivado_hls run.tcl

.PHONY:clean hlsclean
hlsclean:
	rm -rf vivado_hls.log hls.prj
clean: hlsclean
	rm -f main.exe
