#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /home/gc535/ece5775/final_proj/v3/craft-bnn-master/baseline_cpp/baseline/_sds/vhls/top/solution/.autopilot/db/a.g.bc ${1+"$@"}
