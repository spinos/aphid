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
template<int n, typename T> class DictionaryMachine;

class LfWorld  {

	LfParameter * m_param;
    DictionaryMachine<4, float> * m_machine;

public:

	LfWorld(LfParameter * param);
	virtual ~LfWorld();
	
	const LfParameter * param() const;

	void initDictionary();
	void dictionaryAsImage(unsigned * imageBits, int imageW, int imageH);
	void fillSparsityGraph(unsigned * imageBits, int iLine, int imageW, unsigned fillColor);
	void preLearn();
	void learn(const ExrImage * image, int iPatch0, int iPatch1);
	void updateDictionary(const ExrImage * image, int t);
	void beginPSNR();
	void computeError(const ExrImage * image, int iPatch);
	void endPSNR(float * result);
	float computePSNR(const ExrImage * image, int iImage);
	void cleanDictionary();
	void recycleData();
	
	static void testLAR();
protected:
	
private:
	void fillPatch(unsigned * dst, float * color, int s, int imageW, int rank = 3);
	void raiseLuma(unsigned * crgb);
};
}

#endif        //  #ifndef LFWORLD_H

