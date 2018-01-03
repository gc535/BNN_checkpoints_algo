set rootdir $::env(CRAFT_BNN_ROOT)
source $rootdir/sdsoc-example/opt.tcl

set_directive_interface -mode ap_fifo "top" in
set_directive_interface -mode ap_fifo "top" out
set_directive_latency -min 1 top
