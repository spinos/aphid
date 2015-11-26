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
	DenseMatrix<float> * m_D;
	DenseMatrix<float> * m_G;
public:

	LfWorld(LfParameter * param);
	virtual ~LfWorld();
	
	const LfParameter * param() const;

	void initDictionary();
	void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH);
	
protected:
	
private:
	void cleanDictionary();
	void fillPatch(unsigned * dst, float * color, int s, int imageW, int rank = 3);
};
}

