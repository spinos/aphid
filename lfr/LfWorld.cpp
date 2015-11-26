/*
 *  LfWorld.cpp
 *  
 *
 *  Created by jian zhang on 11/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "LfWorld.h"
#include "linearMath.h"
#include "regr.h"
/// f2c macros conflict
#define _WIN32
#include <zEXRImage.h>

namespace lfr {

LfWorld::LfWorld(LfParameter * param) 
{
	m_param = param;
	m_D = new DenseMatrix<float>(param->dimensionOfX(), 
										param->dictionaryLength() );
	m_G = new DenseMatrix<float>(param->dictionaryLength(), 
										param->dictionaryLength() );
}

LfWorld::~LfWorld() {}

const LfParameter * LfWorld::param() const
{ return m_param; }

void LfWorld::initDictionary()
{
	const int n = m_param->numPatches();
	const int k = m_param->dictionaryLength();
	const int s = m_param->atomSize();
	int i, j;
	for(i=0;i<k;i++) {
/// init D with random signal 
        float * d = m_D->column(i);
        ZEXRImage * img = m_param->openImage(m_param->randomImageInd());
        // if(!img) std::cout<<" null_img ";
        
        img->getTile1(d, rand(), s);
	}
	
	m_D->normalize();
	m_D->AtA(*m_G);
	cleanDictionary();
}

void LfWorld::dictionaryAsImage(unsigned * imageBits, int imageW, int imageH)
{
    const int s = m_param->atomSize();
	const int dimx = imageW / s;
	const int dimy = imageH / s;
	const int k = m_param->dictionaryLength();
	int i, j;
	unsigned * line = imageBits;
	for(j=0;j<dimy;j++) {
		for(i=0;i<dimx;i++) {
			const int ind = dimx * j + i;
			if(ind < k) {
			    float * d = m_D->column(ind);
			    fillPatch(&line[i * s], d, s, imageW);
			}
		}
		line += imageW * s;
	}
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
    const int s = m_param->atomSize();
	int i, j, l;
	for (i = 0; i<k; ++i) {
/// lower part of G
		for (j = i; j<k; ++j) {
			bool toClean = false;
			if(j==i) {
/// diagonal part
				toClean = absoluteValue<float>( m_G->column(i)[j] ) < 1e-4;
			}
			else {
				toClean = ( absoluteValue<float>( m_G->column(i)[j] ) / sqrt( m_G->column(i)[i] * m_G->column(j)[j]) ) > 0.9999;
			}
			if(toClean) {
/// D_j <- randomly choose signal element
				DenseVector<float> dj(m_D->column(j), m_D->numRows() );
				
				ZEXRImage * img = m_param->openImage(m_param->randomImageInd());
                img->getTile1(dj.raw(), rand(), s);
                
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

}