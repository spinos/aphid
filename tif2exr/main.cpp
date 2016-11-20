/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 1/11/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>
#include <string>
#include <tiffio.h>
#include <OpenEXR/half.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImfArray.h>

namespace IMF = OPENEXR_IMF_NAMESPACE;
using namespace IMF;
using namespace IMATH_NAMESPACE;

int convertImage( TIFF *in )
{
	uint32* raster;			/* retrieve RGBA image */
    uint32  width, height;		/* image width & height */
    uint32  col, row;
	uint32 rowsperstrip;
	tsample_t samplesperpixel;
	tsample_t bitspersample;
        
    TIFFGetField(in, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(in, TIFFTAG_IMAGELENGTH, &height);
	TIFFGetField(in, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel);
	TIFFGetField(in, TIFFTAG_BITSPERSAMPLE, &bitspersample);
	
	if( !TIFFGetField(in, TIFFTAG_ROWSPERSTRIP, &rowsperstrip) ) {
        TIFFError(TIFFFileName(in), "Source image not in strips");
        return (0);
    }
				
    raster = (uint32*)_TIFFmalloc(width * height * sizeof (uint32));
    if (raster == 0) {
        TIFFError(TIFFFileName(in), "No space for raster buffer");
        return (0);
    }
	
	if (!TIFFReadRGBAImageOriented(in, width, height, raster,
                                   ORIENTATION_TOPLEFT, 0)) {
        TIFFError(TIFFFileName(in), "failed to read rgba image");
        _TIFFfree(raster);
        return (0);
    }
	
	Array2D<half> rPixels(width, height);
	Array2D<half> gPixels(width, height);
	Array2D<half> bPixels(width, height);
	Array2D<half> aPixels(width, height);
	
/// ABGR packed into 32-bit
	const float oneOver256 = .00390256f;
	const uint32 rMask = (1<<24) - 1;
	const uint32 gMask = (1<<16) - 1;
	for(row = 0; row<height;++row) {
		for(col = 0; col<width;++col) {
			uint32 d = raster[row*width+col];
			bPixels[row][col] = oneOver256 * ((d & rMask)>>16);
			gPixels[row][col] = oneOver256 * ((d & gMask)>>8);
			rPixels[row][col] = oneOver256 * (d & 255);
			aPixels[row][col] = oneOver256 * (d>>24);
		}
	}
	
	Header header(width, height);
	header.channels().insert("R", Channel(HALF) );
	header.channels().insert("G", Channel(HALF) );
	header.channels().insert("B", Channel(HALF) );
	header.channels().insert("A", Channel(HALF) );
	
	std::string outFilename(TIFFFileName(in) );
	outFilename += ".exr";
	OutputFile out(outFilename.c_str(), header);
	
	FrameBuffer frameBuffer;
	frameBuffer.insert("R",
						Slice(HALF,
							(char *) &rPixels[0][0],
							sizeof (half) * 1,    // xStride 
							   sizeof (half) * width) );
							   
	frameBuffer.insert("G",
						Slice(HALF,
							(char *) &gPixels[0][0],
							sizeof (half) * 1,    // xStride 
							   sizeof (half) * width) );
							   
	frameBuffer.insert("B",
						Slice(HALF,
							(char *) &bPixels[0][0],
							sizeof (half) * 1,    // xStride 
							   sizeof (half) * width) );
							   
	frameBuffer.insert("A",
						Slice(HALF,
							(char *) &aPixels[0][0],
							sizeof (half) * 1,    // xStride 
							   sizeof (half) * width) );
							   
	out.setFrameBuffer (frameBuffer); 
    out.writePixels (height);
	
	std::cout<<"\n exr file "<<outFilename;
		
	_TIFFfree( raster );
	return 1;
}

int main(int argc, char * const argv[])
{
	if(argc < 2) {
		std::cout<<" tif2exr requires input filename\n";
		return 1;
	}
	
	std::cout<<" tif2exr convert tiff image file "<<argv[1]<<"\n";
	
	TIFF* tif = TIFFOpen(argv[1], "r");
	
	if(!tif) {
		std::cout<<" tif2exr cannot open tiff image file "<<argv[1]<<"\n";
		return 1;
	}
	
	long flags = 0;
	//flags |= TIFFPRINT_COLORMAP;
	//flags |= TIFFPRINT_STRIPS;
	
	do {
		TIFFPrintDirectory(tif, stdout, flags);
		
		if(TIFFIsTiled(tif))
			std::cout<<"\n image is tiled ";
	
		convertImage(tif);
		
	} while (TIFFReadDirectory(tif));

    TIFFClose(tif);
	
	std::cout<<"\n end of tif2exr\n";
	return 0;
}
