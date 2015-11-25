/*
 *  LfWorld.cpp
 *  
 *
 *  Created by jian zhang on 11/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "LfWorld.h"
#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include "linearMath.h"
#include "regr.h"
/// f2c macros conflict
#define _WIN32
#include <zEXRImage.h>
#include <OpenEXR/ImathLimits.h>

namespace lfr {

LfParameter::LfParameter(int argc, char *argv[])
{
	std::cout<<"\n exr limit "<<Imath::limits<int>::min();
	std::cout<<"\n test min "<<std::min<int>(-99, -98);
	std::cout<<"\n test abs "<<std::abs<int>(-75);
	std::cout<<"\n lfr (Light Field Research) version 20151122";
	m_isValid = false;
	m_atomSize = 10;
	m_dictionaryLength = 1024;
	bool foundImages = false;
	if(argc == 1) {
		m_isValid = searchImagesIn("./");
	}
	else {
		int i = 1;
		for(;i<argc;++i) {
			if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
				PrintHelp();
			}
			if(strcmp(argv[i], "-as") == 0 || strcmp(argv[i], "--atomSize") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --atomSize value is not set";
					break;
				}
				try {
					m_atomSize = boost::lexical_cast<int>(argv[i+1]);
					if(m_atomSize < 10) {
						std::cout<<"\n bad --atomSize value (< 10)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --atomSize value "<<argv[i+1];
					break;
				}
			}
			if(strcmp(argv[i], "-dl") == 0 || strcmp(argv[i], "--dictionaryLength") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --dictionaryLength value is not set";
					break;
				}
				try {
					m_dictionaryLength = boost::lexical_cast<int>(argv[i+1]);
					if(m_dictionaryLength < 1024) {
						std::cout<<"\n bad --dictionaryLength value (< 1024)";
						break;
					}
				}
				catch(const boost::bad_lexical_cast &) {
					std::cout<<"\n bad --dictionaryLength value "<<argv[i+1];
					break;
				}
			}
			if(i==argc-1) {
				if(boost::filesystem::exists(argv[i])) {
					m_imageNames.push_back(argv[i]);
					std::cout<<"\n found image "<<argv[i];
					m_isValid = true;
				}
				else 
					std::cout<<"\n image doesn't exist "<<argv[i];
			}
		}
	}
	if(m_isValid) {
		std::cout<<"\n atom size "<<m_atomSize;
		std::cout<<"\n dictionary length "<<m_dictionaryLength;
		countPatches();
		m_isValid = m_numPatches > 0;
	}
}

LfParameter::~LfParameter() {}

bool LfParameter::isValid() const
{ return m_isValid; }

bool LfParameter::searchImagesIn(const char * dirname)
{
	m_imageNames.clear();
	if(!boost::filesystem::exists(dirname)) {
		std::cout<<"\n dir doesn't exist "<<dirname;
		return false;
	}
	std::cout<<"\n searching images in dir "<<dirname<<" ...";
	
	const std::string ext(".exr");
	boost::filesystem::path head_path(dirname);
	boost::filesystem::directory_iterator end_iter;
	for ( boost::filesystem::directory_iterator itdir( head_path );
		  itdir != end_iter;
		  ++itdir ) {
		if ( boost::filesystem::is_regular_file( itdir->status() ) ) {
			const std::string fn = itdir->path().filename().string();
			if(!boost::algorithm::starts_with(fn, ".")) {
			    
				std::string fileext = boost::filesystem::extension(itdir->path().string());
				boost::algorithm::to_lower(fileext);
				if(fileext == ext) {
					m_imageNames.push_back( boost::filesystem::basename(itdir->path()) + ".exr");
				}
			}
		}
	}
	
	std::cout<<" found "<<m_imageNames.size();
	return true;
}

void LfParameter::countPatches()
{
	m_numPatches = 0;
	std::vector<std::string >::const_iterator it = m_imageNames.begin();
	for(; it!=m_imageNames.end();++it) {
		std::string fn = *it;
		
		ZEXRImage img;
		if(img.open(fn.c_str()))
			m_numPatches += (img.getWidth() / m_atomSize) * (img.getHeight() / m_atomSize);
		else 
			std::cout<<"\n cannot open exr "<<fn;
	}
	std::cout<<"\n num patch "<<m_numPatches;
}

int LfParameter::atomSize() const
{ return m_atomSize; }

int LfParameter::dictionaryLength() const
{ return m_dictionaryLength; }

std::string LfParameter::imageName(int i) const
{ return m_imageNames[i]; }

int LfParameter::numPatches() const
{ return m_numPatches; }

int LfParameter::dimensionOfX() const
{ return m_atomSize * m_atomSize * 3; }

int LfParameter::randomImageInd() const
{ 
	if(m_imageNames.size() < 2) return 0; 
	return rand() % m_imageNames.size();
}

void LfParameter::PrintHelp()
{
	std::cout<<"\n lfr (Light Field Research) version 20151122"
	<<"\nUsage:\n lfr [option] [file]"
	<<"\nDescription:\n lfr learns the underlying pattern of input images."
	<<"\n Input file must be image of OpenEXR format. If no file is provided,"
	<<"\n current dir will be searched for any file with name ending in .exr."
	<<"\nOptions:\n -as or --atomSize    integer    size of image atoms, no less than 10"
	<<"\n -dl or --dictionaryLength    integer    length of dictionary, no less than 1024"
	<<"\n -h or --help    print this information"
	<<"\n";
}


LfWorld::LfWorld(const LfParameter & param) 
{
	m_param = &param;
	m_D = new DenseMatrix<float>(param.dimensionOfX(), 
										param.dictionaryLength() );
	m_G = new DenseMatrix<float>(param.dictionaryLength(), 
										param.dictionaryLength() );
										
	int i = 0;
	for(;i<MAX_NUM_OPENED_IMAGES;++i) {
		m_openedImages[i]._ind = -1;
		m_openedImages[i]._image = NULL;
	}
	m_currentImage = 0;
}

LfWorld::~LfWorld() {}

const LfParameter * LfWorld::param() const
{ return m_param; }

void LfWorld::fillDictionary(unsigned * imageBits, int imageW, int imageH)
{
	ZEXRImage img;
	int imgI = m_param->randomImageInd();
	std::string fn = m_param->imageName(imgI);
	img.open(fn.c_str());
	int preImgI = imgI;
	
	const int n = m_param->numPatches();
	const int k = m_param->dictionaryLength();
	const int s = m_param->atomSize();
	const int dimx = imageW / s;
	const int dimy = imageH / s;
	const int m = m_param->dimensionOfX();
	
	int i, j;
	unsigned * line = imageBits;
	for(j=0;j<dimy;j++) {
		for(i=0;i<dimx;i++) {
			const int ind = dimx * j + i;
			if(ind < k) {
/// init D with random signal 
				float * d = m_D->column(ind);
				imgI = m_param->randomImageInd();
				if(preImgI != imgI) {
					fn = m_param->imageName(imgI);
					img.open(fn.c_str());
					preImgI = imgI;
				}
				
				img.getTile(d, rand(), s);
				
				fillPatch(&line[i * s], d, s, imageW);
			}
			else {
				
			}
		}
		line += imageW * s;
	}
	m_D->normalize();
	m_D->AtA(*m_G);
	cleanDictionary();
}

void LfWorld::fillPatch(unsigned * dst, float * color, int s, int imageW, int rank)
{
	int i, j, k;
	unsigned * line = dst;
	for(j=0;j<s; j++) {
		for(i=0; i<s; i++) {
			unsigned v = 255<<24;
			for(k=0;k<rank;k++) {				
				unsigned rgb = 255 * color[(j * s + i) * rank + k];
				rgb = std::min<unsigned>(rgb, 255);
				v = v | ( rgb << ((2-k) << 3) );
			}
			line[i] = v;
		}
		line += imageW;
	}
}

void LfWorld::cleanDictionary()
{
	const int n = m_param->numPatches();
	const int k = m_D->numColumns();
	int i, j, l;
	for (i = 0; i<k; ++i) {
/// lower part of G
		for (j = i; j<k; ++j) {
			bool toClean = false;
			if(j==i) {
/// diagonal part
				toClean = std::abs<float>( m_G->column(i)[j] ) < 1e-4;
			}
			else {
				toClean = ( std::abs<float>( m_G->column(i)[j] ) / sqrt( m_G->column(i)[i] * m_G->column(j)[j]) ) > 0.9999;
			}
			if(toClean) {
/// D_j <- randomly choose signal element
				DenseVector<float> dj(m_D->column(j), m_D->numRows() );
				int rt = rand();
				
				dj.normalize();
/// G_j <- D^t * D_j
				DenseVector<float> gj(m_G->column(j), k);
				m_D->multTrans(gj, dj);
/// copy to diagonal line of G
				for (l = 0; l<k; ++l)
					m_G->column(l)[j] = m_G->column(j)[l];
			}
		}
	}
	m_G->addDiagonal(1e-10);
}

bool LfWorld::isImageOpened(const int ind, ZEXRImage * img) const
{
	int i = 0;
	for(;i<MAX_NUM_OPENED_IMAGES;++i) {
		if(m_openedImages[i]._ind < 0) break;
		
		if( m_openedImages[i]._ind == ind ) {
			img = m_openedImages[i]._image;
			return true;
		}
	}
	return false;
}

void LfWorld::opendImage(const int ind, ZEXRImage * img)
{
	ImageInd & imgind = m_openedImages[m_currentImage];
	if(imgind._ind < 0) {
		imgind._image = new ZEXRImage;
	}
	imgind._ind = ind;
	imgind._image->open(m_param->imageName(ind));
	img = imgind._image;
	
	m_currentImage = (m_currentImage + 1) % MAX_NUM_OPENED_IMAGES;
}

}