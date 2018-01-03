#=========================================================================
# hls.tcl
#=========================================================================

set top "top"
set cflags "-DHLS_COMPILE -O3 -std=c++0x -I../utils -I../minizip"
set tbflags "-DHLS_COMPILE -O3 -std=c++0x -I../utils -I../minizip -lminizip -lcrypto -lz"
set utils "../utils/Common.cpp ../utils/DataIO.cpp ../utils/ParamIO.cpp ../utils/ZipIO.cpp ../utils/Timer.cpp"

open_project hls.prj

set_top $top

add_files Cifar10.cpp -cflags $cflags
add_files -tb cifar10_test_accuracy.cpp -cflags $tbflags
add_files -tb $utils -cflags $tbflags

open_solution "solution1" -reset

set_part {xc7z020clg484-1}
create_clock -period 10

config_rtl -reset state

csim_design

csynth_design
#cosim_design -rtl verilog -trace_level all -argv "1"

# concatenate all the verilog files into a single file
#set fout [open ./$top.v w]
#fconfigure $fout -translation binary
#set vfiles [glob -nocomplain -type f "$top.prj/solution1/syn/verilog/*.v"]
#foreach vfile $vfiles {
#  set fin [open $vfile r]
#  fconfigure $fin -translation binary
#  fcopy $fin $fout
#  close $fin
#}

# add verilator lint off and on pragmas
#seek $fout 0
#puts -nonewline $fout "/* verilator lint_off WIDTH */\n\n//"
#seek $fout -1 end
#puts $fout "\n\n/* lint_on */"
#
#close $fout

exit
