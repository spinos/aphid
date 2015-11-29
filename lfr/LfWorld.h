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

public:

	LfWorld(LfParameter * param);
	virtual ~LfWorld();
	
	const LfParameter * param() const;

	void initDictionary();
	void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH);
	void fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor);
	void preLearn();
	void learn(int iImage, int iPatch);
	void updateDictionary();
	static void testLAR();
protected:
	
private:
	void cleanDictionary();
	void fillPatch(unsigned * dst, float * color, int s, int imageW, int rank = 3);
	void raiseLuma(unsigned * crgb);
};
}

#endif        //  #ifndef LFWORLD_H

