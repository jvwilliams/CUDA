// The CUDA routines below are by Danny Ruijters, adapted by Jonathan Williams.
// External files also required and included.

/*--------------------------------------------------------------------------*\
Copyright (c) 2008-2010, Danny Ruijters. All rights reserved.
http://www.dannyruijters.nl/cubicinterpolation/
This file is part of CUDA Cubic B-Spline Interpolation (CI).

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of the copyright holders nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are
those of the authors and should not be interpreted as representing official
policies, either expressed or implied.

When using this code in a scientific project, please cite one or all of the
following papers:
*  Daniel Ruijters and Philippe Thévenaz,
   GPU Prefilter for Accurate Cubic B-Spline Interpolation, 
   The Computer Journal, vol. 55, no. 1, pp. 15-20, January 2012.
   http://dannyruijters.nl/docs/cudaPrefilter3.pdf
*  Daniel Ruijters, Bart M. ter Haar Romeny, and Paul Suetens,
   Efficient GPU-Based Texture Interpolation using Uniform B-Splines,
   Journal of Graphics Tools, vol. 13, no. 4, pp. 61-69, 2008.
\*--------------------------------------------------------------------------*/



#include <stdio.h>
#include <cubicTex2D.cu>
#include "math_func.cu"
#include <cubicPrefilter2D.cu>

texture<float, 2, cudaReadModeElementType> coeffs;  //2D texture

//--------------------------------------------------------------------------
// CUDA kernel. This is what is actually run on the GPU.
// GPUs are broken down in to streaming multiprocessors, each one of which 
// contains multiple execution units, registers, memory/caches, etc.
// Logically, CUDA programs are broken down into units called blocks. 
// A given block contains multiple threads that are executed simultaneously
// each one on an execution unit),  and a given block must fit into a single
// streaming multiprocessor.
// Part of the job of a streaming multiprocessor is to schedule the
// execution of blocks. So a given CUDA kernel can contain many more threads
// than a GPU has execution units.
//--------------------------------------------------------------------------

// deviceImagePtrNew: the output array
// width: the widht of the raster
// mPlyer: the interpolation multiplier
// rasMax, rasMin: limits on the output value

__global__ void interpolate_kernel(float* deviceImagePtrNew, uint width, float mPlyer, float rasMax, float rasMin)
{
	// Find the index i of the array element this thread will put its output into.
	// The threads in this kernel are organized into two dimensions, even though
	// the output array is one-dimensional.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;

	// mPlyer is used to subdivide the existing raster into smaller units for interpolation.
	float x0 = (float)x;
	float y0 = (float)y;
	float x1 = x0/mPlyer;
	float y1 = y0/mPlyer;


	deviceImagePtrNew[i] = fminf(fmaxf(cubicTex2D(coeffs, x1, y1),rasMin),rasMax);
}


//--------------------------------------------------------------------------
// Copy floating point data from and to the GPU
//--------------------------------------------------------------------------

//! Allocate GPU memory and copy a voxel volume from CPU to GPU memory
//! @return the pitched pointer to the GPU copy of the voxel volume
//! @param host  pointer to the voxel volume in CPU (host) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
extern "C" cudaPitchedPtr CopyVolumeHostToDevice(const float* hostImagePtr, uint width, uint height, uint depth)
{
	cudaPitchedPtr deviceImagePPtr = {0};
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMalloc3D(&deviceImagePPtr, extent);
	cudaMemcpy3DParms p = {0};
	p.srcPtr = make_cudaPitchedPtr((void*)hostImagePtr, width * sizeof(float), width, height);
	p.dstPtr = deviceImagePPtr;
	p.extent = extent;
	p.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&p);
	return deviceImagePPtr;
}

//! Copy a voxel volume from GPU to CPU memory, and free the GPU memory
//! @param host  pointer to the voxel volume copy in CPU (host) memory
//! @param device  pitched pointer to the voxel volume in GPU (device) memory
//! @param width   volume width in number of voxels
//! @param height  volume height in number of voxels
//! @param depth   volume depth in number of voxels
//! @note The \host CPU memory should be pre-allocated
extern "C" void CopyVolumeDeviceToHost(float* hostImagePtr, const cudaPitchedPtr deviceImagePPtr, uint width, uint height, uint depth)
{
	const cudaExtent extent = make_cudaExtent(width * sizeof(float), height, depth);
	cudaMemcpy3DParms p = {0};
	p.srcPtr = deviceImagePPtr;
	p.dstPtr = make_cudaPitchedPtr((void*)hostImagePtr, width * sizeof(float), width, height);
	p.extent = extent;
	p.kind = cudaMemcpyDeviceToHost;
	cudaMemcpy3D(&p);
	cudaFree(deviceImagePPtr.ptr);  //free the GPU volume
}

// This function executes the first two of three kernels on the GPU.
// The prefilter is used to alter the input array such that the output interpolated array always goes
// through the original raster values. If we didn't do this, the output would be smoothed overall.
// Using the prefilter ends up sharpening the output.
template<class floatN>
extern void CubicBSplinePrefilter2D(cudaPitchedPtr deviceImagePPtr, uint width, uint height)
{
	dim3 dimBlockX(max(min(PowTwoDivider(height), 64), 1024));
	dim3 dimGridX(height / dimBlockX.x);
	SamplesToCoefficients2DX<floatN><<<dimGridX, dimBlockX>>>(deviceImagePPtr.ptr, deviceImagePPtr.pitch, width, height);

	dim3 dimBlockY(max(min(PowTwoDivider(width), 64), 1024));
	dim3 dimGridY(width / dimBlockY.x);
	SamplesToCoefficients2DY<floatN><<<dimGridY, dimBlockY>>>(deviceImagePPtr.ptr, deviceImagePPtr.pitch, width, height);
}

// This algorithm makes use of the texture memory and manipulation capabilities of the GPU.
// initTexture takes the input finds relevant spline interpolation coefficients, and puts them
// into texture memory on the GPU.
extern "C" void initTexture(cudaPitchedPtr deviceImagePPtr, uint width, uint height)
{
	// Create the B-spline coefficients texture
	cudaChannelFormatDesc channelDescCoeff = cudaCreateChannelDesc<float>();
	cudaArray *coeffArray = 0;
	cudaMallocArray(&coeffArray, &channelDescCoeff, width, height);
	cudaMemcpy2DToArray(coeffArray, 0, 0, deviceImagePPtr.ptr, deviceImagePPtr.pitch, width * sizeof(float), height, cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(coeffs, coeffArray, channelDescCoeff);
	coeffs.normalized = false;  // access with normalized texture coordinates
	coeffs.filterMode = cudaFilterModeLinear;
}

// The interpolate function does the work of allocating output memory on the GPU and executing the
// kernel.
extern "C" cudaPitchedPtr interpolate(uint width, uint height, float mPlyer, float rasMax, float rasMin)
{

	// Allocate the output image
	float* deviceImagePtrNew;
	cudaMalloc((void**)&deviceImagePtrNew, width * height * sizeof(float));


	// here we calculate the appropriate number of threads (in two dimensions) per block.
	// A given generation of GPU has a limit on the number of threads per block.
	
	int bSizeX = min(PowTwoDivider(width), 16);
	int bSizeY = min(PowTwoDivider(height), 16);
	if (bSizeX * bSizeY > 1024) {
		bSizeX = 32;
		bSizeY = 32;
	}
	dim3 blockSize(bSizeX, bSizeY);
	dim3 gridSize(width / blockSize.x, height / blockSize.y);
	
	// Call the CUDA execution kernel
	interpolate_kernel<<<gridSize, blockSize>>>(deviceImagePtrNew, width, mPlyer, rasMax, rasMin);

	//return a pointer to the resulting output on the GPU
	return make_cudaPitchedPtr(deviceImagePtrNew, width * sizeof(float), width, height);
}

extern "C" void MyCubicBSplinePrefilter2D(cudaPitchedPtr image, uint width, uint height)
{
	return CubicBSplinePrefilter2D((float*)image.ptr, (uint)image.pitch, width, height);
}