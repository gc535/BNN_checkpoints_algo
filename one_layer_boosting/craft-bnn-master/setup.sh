#!/usr/bin/env bash

export CPATH=$CPATH:/work/zhang/common/tools/xilinx/SDSoC/2016.1/Vivado_HLS/2016.1/include
source /work/zhang/common/tools/xilinx/SDSoC/2016.1/settings64.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export CRAFT_BNN_ROOT=$DIR
