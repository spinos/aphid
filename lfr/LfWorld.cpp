/*
 *  LfWorld.cpp
 *  
 *
 *  Created by jian zhang on 11/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "LfWorld.h"
#include "regr.h"
#include "psnr.h"
/// f2c macros conflict
#define _WIN32
#include <ExrImage.h>
#include <MersenneTwister.h>


namespace lfr {

LfWorld::LfWorld(LfParameter * param) 
{
	m_param = param;
	const int m = param->dimensionOfX();
	const int p = param->dictionaryLength();
	m_D = new DenseMatrix<float>(m, p);
	m_G = new DenseMatrix<float>(p, p);
	m_A = new DenseMatrix<float>(p, p);
	m_B = new DenseMatrix<float>(m, p);
	m_lar = new LAR<float>(m_D, m_G);
	m_y = new DenseVector<float>(m);
	m_beta = new DenseVector<float>(p);
	m_ind = new DenseVector<int>(p);
	m_errorCalc = new Psnr<float>(m_D);
	m_batchA = new DenseMatrix<float>(p, p);
	m_batchB = new DenseMatrix<float>(m, p);
}

LfWorld::~LfWorld() {}

const LfParameter * LfWorld::param() const
{ return m_param; }

void LfWorld::initDictionary()
{
	const int k = m_param->dictionaryLength();
	const int s = m_param->atomSize();
	int i, j;
	for(i=0;i<k;i++) {
/// init D with random signal 
        float * d = m_D->column(i);
        ExrImage * img = m_param->openImage(m_param->randomImageInd());
        // if(!img) std::cout<<" null_img ";
        
        img->getTile(d, rand(), s);
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
    const int stride = s * s;
	int crgb[3];
	int i, j, k;
	unsigned * line = dst;
	for(j=0;j<s; j++) {
		for(i=0; i<s; i++) {
			unsigned v = 255<<24;
			for(k=0;k<rank;k++) {				
				crgb[k] = 8 + 500 * color[(j * s + i) + k * stride];
				crgb[k] = std::min<int>(crgb[k], 255);
				crgb[k] = std::max<int>(crgb[k], 0);
			}
			v = v | ( crgb[0] << 16 );
			v = v | ( crgb[1] << 8 );
			v = v | ( crgb[2] );
			line[i] = v;
		}
		line += imageW;
	}
}

void LfWorld::cleanDictionary()
{
    const int k = m_D->numColumns();
    const int s = m_param->atomSize();
	int i, j, l;
	for (i = 0; i<k; ++i) {
/// lower part of G
		for (j = i; j<k; ++j) {
			bool toClean = false;
			if(j==i) {
/// diagonal part
				toClean = absoluteValue<float>( m_G->column(i)[j] ) < 1e-7;
			}
			else {
				float ab = m_G->column(i)[i] * m_G->column(j)[j];
				toClean = ( absoluteValue<float>( m_G->column(i)[j] ) / sqrt( ab ) ) > 0.9999;
			}
			if(toClean) {
/// D_j <- randomly choose signal element
				DenseVector<float> dj(m_D->column(j), m_D->numRows() );
				
				ExrImage * img = m_param->openImage(m_param->randomImageInd());
                img->getTile(dj.raw(), rand(), s);
                
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
	m_G->addDiagonal(1e-8);
}

void LfWorld::preLearn()
{
	m_A->setZero();
	m_A->addDiagonal(1e-5);
	m_B->copy(*m_D);
	m_B->scale(1e-7);
	
	m_batchA->setZero();
	m_batchB->setZero();
}

void LfWorld::learn(const ExrImage * image, int iPatch)
{
	const int k = m_D->numColumns();
	const int s = m_param->atomSize();
	
	image->getTile(m_y->raw(), iPatch, s);

	m_lar->lars(*m_y, *m_beta, *m_ind, 0.0);
	
	int nnz = 0;
	int i=0;
	for(;i<k;++i) {
		if((*m_ind)[i] < 0) break;
		nnz++;
	}
	if(nnz < 1) return;
	
	sort<float, int>(m_ind->raw(), m_beta->raw(), 0, nnz-1);
	
/// A <- A + beta * beta^t
	m_batchA->rank1Update(*m_beta, *m_ind, nnz);
/// B <- B + y * beta^t
	m_batchB->rank1Update(*m_y, *m_beta, *m_ind, nnz);
}

void LfWorld::updateDictionary(int niter)
{
/// combine a batch
    m_batchA->scale(1.0/256);
    m_batchB->scale(1.0/256);
/// reduce A increases chance to clean an atom
/// blindly select a patch can be equally bad, lower than 0.9 is too random
/// lesser scaling after more loops to reduce cleaning
    float sc = 0.98;
    //float sc = float(niter+1000 - 30)/float(niter+1000);
    m_A->scale(sc);
    m_B->scale(sc);
    m_A->add(*m_batchA);
    m_B->add(*m_batchB);
    m_batchA->setZero();
    m_batchB->setZero();
    
	const int p = m_D->numColumns();
	DenseVector<float> ui(m_D->numRows());
	int i, j;
/// repeat ?
//	for (j = 0; j<1; ++j) {
		for (i = 0; i<p; ++i) {
			const float Aii = m_A->column(i)[i];
			if (Aii > 1e-6) {
				DenseVector<float> di(m_D->column(i), m_D->numRows());
				DenseVector<float> ai(m_A->column(i), m_A->numRows());
				DenseVector<float> bi(m_B->column(i), m_B->numRows());
/// ui <- (bi - D * ai) / Aii + di
				m_D->mult(ui, ai, -1.0f);
				ui.add(bi);
				ui.scale(1.0f/Aii);
				ui.add(di);
/// di <- ui / max(|| ui ||, 1)				
				float unm = ui.norm();
				if(unm > 1.0) 
					ui.scale(1.f/unm);
				
				m_D->copyColumn(i, ui.v());
		   }
		   else {
				DenseVector<float> di(m_D->column(i), m_D->numRows());
				di.setZero();
		   }
		}		
//	}
	
	m_D->normalize();
	m_D->AtA(*m_G);
	cleanDictionary();
}

void LfWorld::fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor)
{
	DenseVector<unsigned> scanline(&imageBits[iLine * imageW], imageW);
	scanline.setZero();
	const int k = m_param->dictionaryLength();
	const float ratio = (float)k / imageW;
			
	int i = 0;
	for(;i<imageW;++i) {
		if((*m_ind)[i*ratio] < 0) break;
		scanline[i] = fillColor;
	}
}

void LfWorld::beginPSNR()
{ m_errorCalc->reset(); }

void LfWorld::computeError(const ExrImage * image, int iPatch)
{
	const int s = m_param->atomSize();
	image->getTile(m_y->raw(), iPatch, s);
	m_lar->lars(*m_y, *m_beta, *m_ind, 0.0);
	m_errorCalc->add(*m_y, *m_beta, *m_ind);
}

void LfWorld::endPSNR(float * result)
{ *result = m_errorCalc->finish(); }

void LfWorld::testLAR()
{
	std::cout<<"\n test least angle regression";
	MersenneTwister twist(99);
	
	const int p = 200;
	const int m = 11;
	DenseMatrix<float> A(m, p);
	DenseMatrix<float> G(p, p);
	
	float * c0 = A.raw();
	int i, j;
	for(i=0; i<p; i++) {
		for(j=0;j<m;j++) {
			c0[i*m+j] = twist.random() - 0.5;
		}
	}
	
	A.normalize();
	A.AtA(G);
	
	DenseVector<float> y(m);
	y.copyData(A.column(23));
	for(i=0;i<m;i++)
		y.raw()[i] += .03 * (twist.random() - 0.5);
	
	y.scale(10.0);
	
	DenseVector<float> beta(p);
	DenseVector<int> ind(p);
	LAR<float> lar(&A, &G);
	lar.lars(y, beta, ind, 0.0005);
}

void LfWorld::raiseLuma(unsigned * crgb)
{
	int Y = 0.299 * crgb[0] + 0.587 * crgb[1] + 0.114 * crgb[2];
	if(Y < 8) Y = 8;
	int Cb = 128 - 0.168736 * crgb[0] - 0.331264 * crgb[1] + 0.5 * crgb[2];
	int Cr = 128 + 0.5 * crgb[0] - 0.418688 * crgb[1] - 0.081312 * crgb[2];
	crgb[0] = Y + 1.402 * (Cr - 128);
	crgb[1] = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128);
	crgb[2] = Y + 1.772 * (Cb - 128);
}

}