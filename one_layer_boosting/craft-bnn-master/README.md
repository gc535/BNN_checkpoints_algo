# craft-bnn

## params

Use get\_params.sh to download the parameter files.

This directory contains the BNN parameter files. These are stored
in both numpy archives (.npz) which is the original produced by
the training scripts, as well as .zip which can be read by C++.
The easy to use python program npz2zip can do the conversion.

The archives are organized into files. Each file in an archive 
stores one named parameter in one layer. A layer may have multiple
named parameters:
* Conv: W
* Dense: W
* BatchNorm: k, h

## data

Use get\_data.sh to download the data files.

Contains both the cifar10 test data/labels, as well as outputs
of the BNN after each layer. The layer outputs are organized
as a zip archive with a single file (single array).

## python

Contains Python code relevant to working with the parameters. Files
in UpperCase indicate utility classes while files in lower\_case
indicate programs.

InMemoryZip is a class to write zip archives.

FixedPoint is a class to quantize floating to fixed point.

npz2zip is a program to convert a numpy archive to a zip archive,
which can be read by C++ code.

## minizip

This is a library which is needed for the C++ code to access zip files.
Build with "make" in the minizip directory. Install by copying all
header (.h) files in the minizip directory to "$PREFIX/include/minizip"
and all lib (.a) files to "$PREFIX/lib/".

## cpp

Do "make" to build all relevant programs. "make hls" will build the HLS
accelerator.

Currently there are two programs:
  * cifar10\_layer\_test: runs the accelerator code and checks output
  feature maps against that of the Python program.
  * cifar10\_accuracy\_test: runs the accelerator code to classify
  the cifar10 test images and reports the error rate. THe program
  takes about 2 seconds per image and 8 hours for the entire dataset.

The environment variable CRAFT\_BNN\_ROOT should be set to the topmost
directory so the programs can find the params and data. setup.sh can do
this for you.
