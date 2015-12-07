/*
 *  LfWorld.cpp
 *  
 *
 *  Created by jian zhang on 11/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "LfWorld.h"
#include "dctmn.h"
#include <MersenneTwister.h>

namespace lfr {

LfWorld::LfWorld(LfParameter * param) 
{
	m_param = param;
	const int m = param->dimensionOfX();
	const int p = param->dictionaryLength();
	m_machine = new DictionaryMachine<4, float>(m, p);
}

LfWorld::~LfWorld() 
{ delete m_machine; }

const LfParameter * LfWorld::param() const
{ return m_param; }

void LfWorld::initDictionary()
{
    m_machine->initDictionary(m_param);
    /*
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
	cleanDictionary();*/
}

void LfWorld::dictionaryAsImage(unsigned * imageBits, int imageW, int imageH)
{
    m_machine->dictionaryAsImage(imageBits, imageW, imageH, m_param);
    /*
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
	}*/
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
    m_machine->cleanDictionary(m_param);
    /*
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
	*/
}

void LfWorld::preLearn()
{
    m_machine->preLearn();
    /*
	m_A->setZero();
	m_A->addDiagonal(1e-5);
	m_B->copy(*m_D);
	m_B->scale(1e-7);
	
	m_batchA->setZero();
	m_batchB->setZero();*/
}

void LfWorld::learn(const ExrImage * image, int iPatch0, int iPatch1)
{
    m_machine->learn(image, iPatch0, iPatch1);
}

void LfWorld::updateDictionary(const ExrImage * image, int t)
{
    m_machine->updateDictionary(image, t);
    /*
/// combine a batch weighted to smaller step
    m_batchA->scale(1.0/720);
    m_batchB->scale(1.0/720);
/// reduce A increases chance to clean an atom
/// blindly select a patch can be equally bad, lower than 0.9 is too random
/// A accumulated after each iteration, lesser chance to clean
    float sc = 0.93;
    //float sc = float(niter+10000 - 256)/float(niter+10000);
    m_A->scale(sc);
    m_B->scale(sc);
    m_A->add(*m_batchA);
    m_B->add(*m_batchB);
    m_batchA->setZero();
    m_batchB->setZero();
    
	const int p = m_D->numColumns();
	DenseVector<float> ui(m_D->numRows());
	int i;
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
		       if(forceClean) {
		           DenseVector<float> di(m_D->column(i), m_D->numRows());
		           di.setZero();
		       }
		   }
		}		
	
	m_D->normalize();
	m_D->AtA(*m_G);
	cleanDictionary();
	*/
}

void LfWorld::fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor)
{
    m_machine->fillSparsityGraph(imageBits, iLine, imageW, fillColor);
}

void LfWorld::beginPSNR()
{ 
    m_machine->beginPSNR(); 
}

void LfWorld::computeError(const ExrImage * image, int iPatch)
{
    m_machine->computeError(image, iPatch);
}

void LfWorld::endPSNR(float * result)
{ 
    m_machine->endPSNR(result, m_param->totalNumPixels()); 
}

float LfWorld::computePSNR(const ExrImage * image, int iImage)
{
    const int m = m_param->imageNumPatches(iImage);
    const int s = m_param->atomSize();
	return m_machine->computePSNR(image, m, s);
}

void LfWorld::recycleData()
{ m_machine->recycleData(); }

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