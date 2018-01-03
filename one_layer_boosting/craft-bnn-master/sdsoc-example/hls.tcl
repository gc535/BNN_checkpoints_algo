#=========================================================================
# hls.tcl
#=========================================================================

set top "top"
set cflags "-DHLS_COMPILE -O3 -std=c++0x"
set tbflags "-DHLS_COMPILE -O3 -std=c++0x"

open_project hls.prj

set_top $top

add_files Top.cpp -cflags $cflags
add_files -tb test.cpp -cflags $cflags

open_solution "solution1" -reset

set_part {xc7z020clg484-1}
create_clock -period 5

config_rtl -reset state

# Apply optimizations
source opt.tcl

csim_design

csynth_design
cosim_design -rtl verilog -trace_level all

#export_design -evaluate verilog

exit
