# CUDA

This is a small application for opening multiple raster files (any file 
type GDAL can open), carrying out a cubic b-spline interpolation on that
raster, and writing the results.

Very little error checking is done in the files. The main point was getting 
the program to compile and work, and work fast.

As compared with GDALWarp with three CPU threads, this program is more than 
65 times faster in carrying out an eightfold cubic spline interpolation with a 
1440x721 raster when run on two nVidia m2090 GPU's in parallel (two threads 
called by a test PHP script). The GPU test was likely limited by PCI bus 
bandwidth and other factors.

The cubic b-spline interpolation routines are by Danny Ruijters, and adapted here for 
this application. His work and further information may be found at 
http://www.dannyruijters.nl/cubicinterpolation/ .

The program also makes use of the GDAL API. Documentation may be found at 
http://www.gdal.org/gdal_tutorial.html , and more general documentation for GDAL 
may be found here: http://www.gdal.org/index.html

gpuappmf.c : the main c source file.<br>
gpuapp.cu:    the CUDA source file

The remaining files are necessary include files, pared down from the include files by Dr. 
Ruijters.

gpuappmf.c is compiled with gcc, and gpuapp.cu is compiled with nVidia's nvcc compiler. 
The resulting object files are combined with gcc.

Input command line arguments:

0) Program (self), not part of the command line arguments<br>
1) GPU Number<br>
2) Interpolation Resolution Multiplier<br>
3) Raster Maximum Data Value<br>
4) Raster Minimum Data Value<br>
5) Input File Start<br>
6) Input File End<br>
7) File Prefix<br>
8) Input File Extension<br>

Input file names are of the form: 
[prefix]_[file number].[input file extension], e.g. file_1.tif

Output file names are of the form:
[prefix]_[file number]_res.tif, e.g. file_1_res.tif

All output files are in geotiff format by default. The program assumes all input files 
have the same dimensions.


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
