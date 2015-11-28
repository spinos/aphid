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

template<typename T> class DenseMatrix;

class LfWorld  {

	LfParameter * m_param;
/// dictionary
	DenseMatrix<float> * m_D;
/// gram of D
	DenseMatrix<float> * m_G;
/// beta * beta^t
	DenseMatrix<float> * m_A;
/// X * beta^t
	DenseMatrix<float> * m_B;
public:

	LfWorld(LfParameter * param);
	virtual ~LfWorld();
	
	const LfParameter * param() const;

	void initDictionary();
	void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH);
	void preLearn();
	void learn(int iImage, int iPatch);
	
	static void testLAR();
protected:
	
private:
	void updateDictionary();
	void cleanDictionary();
	void fillPatch(unsigned * dst, float * color, int s, int imageW, int rank = 3);
};
}

#endif        //  #ifndef LFWORLD_H

