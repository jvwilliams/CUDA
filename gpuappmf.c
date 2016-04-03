/*===================================================================================================
cudainterp
A small app for b-spline interpolation of a GDAL-readable, single-band raster
Requires GDAL and CUDA libraries (of course).
This application takes arguments that select a sequential range of files, and processes them with the
same interpolation values.
Very little error checking is done in this application. Obviously production apps should include
this; it was assumed in this application that such error checking is done externally, and
not done here to save time.
This application executes on a single GPU, and can be run simultaneously on multiple GPUs by
selecting which GPU to run on (argument 1)
=====================================================================================================
*/

#include "gdal.h"
#include "cpl_conv.h" /* for CPLMalloc() */
#include "cpl_string.h"
/*
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
*/

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif


struct cudaPitchedPtr {
	void* ptr;
	size_t pitch;
	size_t xsize;
	size_t ysize;
};


EXTERN void cudaSetDevice(int deVice);
EXTERN struct cudaPitchedPtr CopyVolumeHostToDevice(float* host, uint width, uint height, uint depth);
EXTERN void CopyVolumeDeviceToHost(float* host, struct cudaPitchedPtr device, uint width, uint height, uint depth);
EXTERN void MyCubicBSplinePrefilter2D(struct cudaPitchedPtr image, uint width, uint height);
EXTERN struct cudaPitchedPtr interpolate(uint width, uint height, float mPlyer, float rasMax, float rasMin);
EXTERN void initTexture(struct cudaPitchedPtr bsplineCoeffs, uint width, uint height);

/*
Arguments:
0) Program (self), not part of the command line arguments
1) GPU Number
2) Interpolation Resolution Multiplier
3) Raster Max Value
4) Raster Minimum Value
5) Input File Start
6) Input File End
7) File Prefix
8) Input File Extension
*/

int main( int argc, char *argv[] )  {
	GDALDriverH	formatDriver;
	double		geoTransform[6];
	GDALDatasetH	datasetIn,datasetOut;
	GDALRasterBandH	bandIn, bandOut;
	int		nBlockXSize, nBlockYSize;
	int i, inXSize, inYSize, outXSize, outYSize;
	int		bGotMin, bGotMax;
	char inFile[30];
	char outFile[30];
	struct		cudaPitchedPtr gpuRefIn, cudaOutput;	
	
	float	*rasterIn = NULL;
	float	*rasterOut = NULL;
	char		**dsCreateOptions = NULL;
	const char	*rasterFormat = "GTiff";
	
	// Process input arguments
	int deVice = strtol(argv[1],NULL,10);
	float mPlyer = strtof(argv[2],NULL);
	float rasMax = strtof(argv[3],NULL);
	float rasMin = strtof(argv[4],NULL);
	int fStart = (int)strtol(argv[5],NULL,10);
	int fEnd = (int)strtol(argv[6],NULL,10);
	const char *preFix = argv[7];
	const char *postFix = argv[8];
	
	if (mPlyer > 30) {
		mPlyer = 30;
	}
	if (mPlyer <= 0) {
		mPlyer = 0.01;
	}
	
	// Choose our GPU (if more than one available). Typically numbered sequentially,
	// starting with "0".
	cudaSetDevice(deVice);
	
	// Initialize GDAL. We use GDAL for input and output because of the wide range of 
	// i/o raster type drivers included, and the ability to read and write geographic
	// characteristics
	GDALAllRegister();
	
	// Loop to process the specified files. Input files are of the form:
	// [prefix]_[file number].[input file extention]
	// Output files are of the form:
	// [prefix]_[file number]_res.tif
	// All output files are in geotiff format.
	for(i = fStart; i <= fEnd; i++){
		// create input and output file names
		sprintf(inFile,"%s_%d.%s",preFix,i,postFix);
		sprintf(outFile,"%s_%d_res.tif",preFix,i);
		
		// Open our input raster file
		datasetIn = GDALOpen( inFile, GA_ReadOnly );
		if( datasetIn == NULL )
		{
			CPLError( CE_Failure, CPLE_AppDefined, "Source file not found" );
			exit(1);
		}
		
		// Open the input file raster band 1 (we expect only one raster band in this
		// application).
  	bandIn = GDALGetRasterBand( datasetIn, 1 );
  	
  	// Check whether this is our first time through the loop. If so, do additional
  	// GDAL initialization and allocate memory for input and output rasters
		if(rasterIn == NULL){
	  	formatDriver = GDALGetDriverByName( rasterFormat );
			inXSize = GDALGetRasterBandXSize( bandIn );
			inYSize = GDALGetRasterBandYSize( bandIn );
			rasterIn = (float *) VSIMalloc(sizeof(float)*inXSize*inYSize);
			outXSize = mPlyer * inXSize;
			outYSize = mPlyer * inYSize;
			rasterOut = (float *) VSIMalloc(sizeof(float)*outXSize*outYSize);
		}
		
		// Create our output data file
		datasetOut = GDALCreate( formatDriver, outFile, outXSize, outYSize, 1, GDT_Float32, dsCreateOptions );
		
		// Various GDAL functions, transferring geographic characteristics from the input
		// raster to the output raster
		GDALSetProjection( datasetOut, GDALGetProjectionRef( datasetIn ) );
		GDALGetGeoTransform( datasetIn, geoTransform );
		geoTransform[1]/=mPlyer;
		geoTransform[5]/=mPlyer;
		GDALSetGeoTransform( datasetOut, geoTransform );
		bandOut = GDALGetRasterBand( datasetOut, 1 );
		GDALSetDescription(bandOut, GDALGetDescription(bandIn));
		
		// Read in the data from the input raster band
  	GDALRasterIO( bandIn, GF_Read, 0, 0, inXSize, inYSize, rasterIn, inXSize, inYSize, GDT_Float32, 0, 0 );
  	
  	// Copy the data to the GPU
		gpuRefIn = CopyVolumeHostToDevice(rasterIn, inXSize, inYSize, 1);
		
		// Prefilter the data
		MyCubicBSplinePrefilter2D(gpuRefIn, inXSize, inYSize);
		
		//Create spline coefficients in GPU texture memory
		initTexture(gpuRefIn, inXSize, inYSize);
		
		// Run the actual interpolation on the GPU
		cudaOutput = interpolate(outXSize, outYSize, mPlyer, rasMax, rasMin);
		
		// Bring the data back from the GPU to here (into rasterOut)
		CopyVolumeDeviceToHost(rasterOut, cudaOutput, outXSize, outYSize, 1);
		
		// Write the data to our output file
 		GDALRasterIO( bandOut, GF_Write, 0, 0, outXSize, outYSize, rasterOut, outXSize, outYSize, GDT_Float32, 0, 0 );
 		
 		// Close the input and output raster files
		GDALClose(datasetIn);
		GDALClose(datasetOut);
	}
	
	// Free up our raster memory
	VSIFree(rasterIn);
	VSIFree(rasterOut);
	return 0;
}