#=========================================================================
# hls.tcl
#=========================================================================

set top "top"

open_project hls.prj

set_top $top

add_files main.cpp -cflags "-O2 -std=c++0x"
add_files -tb main.cpp -cflags "-O2 -std=c++0x"

open_solution "solution1" -reset

set_part {xc7z020clg484-1}
create_clock -period 10

config_rtl -reset state

csim_design

csynth_design
cosim_design -rtl verilog -trace_level all

exit
