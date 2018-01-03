open_project top
set_top top
add_files /home/gc535/ece5775/final_proj/v3/craft-bnn-master/baseline_cpp/baseline/cifar10_test_accuracy.cpp -cflags "-I/home/gc535/ece5775/final_proj/v3/craft-bnn-master/baseline_cpp/utils -I/home/gc535/ece5775/final_proj/v3/craft-bnn-master/baseline_cpp/minizip -Wall -O3 -DHLS_COMPILE -O3 -std=c++0x -lminizip -lcrypto -lz -D __SDSCC__ -m32 -I /opt/xilinx/Xilinx_SDx_2017.2_sdx_0823_1/SDx/2017.2/target/aarch32-linux/include -I/home/gc535/ece5775/final_proj/v3/craft-bnn-master/baseline_cpp/baseline -D __SDSVHLS__ -D __SDSVHLS_SYNTHESIS__ -I /home/gc535/ece5775/final_proj/v3/craft-bnn-master/baseline_cpp/baseline -w"
open_solution "solution" -reset
set_part { xc7z045ffg900-2 }
# synthesis directives
create_clock -period 7.000001
set_clock_uncertainty 27.0%
config_rtl -reset_level low
source /home/gc535/ece5775/final_proj/v3/craft-bnn-master/baseline_cpp/baseline/_sds/vhls/top.tcl
# end synthesis directives
config_rtl -prefix a0_
csynth_design
export_design -ipname top -acc
exit
