# CUDA

This is a small application for opening multiple raster files (any file
type GDAL can open), carrying out a cubic b-spline interpolation on that
raster, and writing the results.

Very little error checking is done in the files.

gpuappmf.c : the main c source file.
gpuapp.cu:    the CUDA source file

gpuappmf.c is compiled with gcc, and gpuapp.cu is compiled with nVidia's nvcc compiler. 
The resulting object files are combined with gcc.

The remaining files are necessary include files.

The cubic b-spline interpolation routines are by Danny Ruijters, and adapted here for this application.

Input command line arguments:

0) Program (self), not part of the command line arguments<br>
1) GPU Number
2) Interpolation Resolution Multiplier
3) Raster Maximum Data Value
4) Raster Minimum Data Value
5) Input File Start
6) Input File End
7) File Prefix
8) Input File Extension

Input file names are of the form: 
[prefix]_[file number].[input file extension], e.g. file_1.tif

Output file names are of the form:
[prefix]_[file number]_res.tif, e.g. file_1_res.tif

All output files are in geotiff format by default.


Requires: GDAL, CUDA, and the following libraries:  libgdal and libstdc++.

Compile instructions and command lines:

nvcc -m64 -O3 -gencode arch=compute_20,code=sm_20 -I/usr/local/cuda-7.5/include -I. -I/usr/local/cuda-7.5/samples/common/inc -I../../code -I../cuda5_fix -o gpuapp.o -c gpuapp.cu

gcc -m64  -O3 -I/usr/local/cuda-7.5/include -I. -I/usr/local/cuda-7.5/samples/common/inc -o cpuappmf.o -c gpuappmf.c

gcc -m64 -o gpuappmf cpuappmf.o gpuapp.o -L/usr/local/cuda-7.5/lib64 -lcudart -L/usr/local/cuda-7.5/samples/common/lib/linux/x86_64
-L/usr/X11R6/lib -I /usr/local/include -L /usr/local/lib -lgdal -lstdc++

NOTE: compiling the cuda code for GPU's with higher compute levels than 2.0 should be done by adjusting "arch=compute_20,code=sm_20" in
the nvcc compile line for optimal code.

If using cuda SDK versions other than 7.5, adjust the compile lines accordingly.

The compilation lines above are specifically for 64 bit code.
