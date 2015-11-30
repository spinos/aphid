#ifndef LFWORLD_H
#define LFWORLD_H

/*
 *  LfWorld.h
 *  
 *
 *  Created by jian zhang on 11/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "LfParameter.h"

namespace lfr {

template<typename T> class DenseVector;
template<typename T> class DenseMatrix;
template<typename T> class LAR;
template<typename T> class Psnr;

class LfWorld  {

	LfParameter * m_param;
/// signal
	DenseVector<float> * m_y;
/// coefficients
	DenseVector<float> * m_beta;
/// sparse indices
	DenseVector<int> * m_ind;
/// dictionary
	DenseMatrix<float> * m_D;
/// gram of D
	DenseMatrix<float> * m_G;
/// beta * beta^t
	DenseMatrix<float> * m_A;
/// X * beta^t
	DenseMatrix<float> * m_B;
/// least angle regression
	LAR<float> * m_lar;
/// peak signal-to-noise ratio
	Psnr<float> * m_errorCalc;
/// by 'inpainting color images in learned dictionary'
/// M = I + gamma / n * K
/// I is 3n-by-3n identity
/// gamma = 5.25
/// n is num pixels in a patch
/// K is 3n-by-3n diagonal by three n-by-n ones
/// y should be (R,G,B)^t of 3n length
/// y^t * M to get average color of signal
    DenseMatrix<float> * m_average;
public:

	LfWorld(LfParameter * param);
	virtual ~LfWorld();
	
	const LfParameter * param() const;

	void initDictionary();
	void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH);
	void fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor);
	void preLearn();
	void learn(const ZEXRImage * image, int iPatch);
	void updateDictionary();
	void beginPSNR();
	void computeError(const ZEXRImage * image, int iPatch);
	void endPSNR(float * result);
	
	static void testLAR();
protected:
	
private:
	void cleanDictionary();
	void fillPatch(unsigned * dst, float * color, int s, int imageW, int rank = 3);
	void raiseLuma(unsigned * crgb);
};
}

#endif        //  #ifndef LFWORLD_H

