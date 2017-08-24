/*
 *  StemBlock.h
 *  
 *
 *  Created by jian zhang on 8/24/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_STEM_BLOCK_H
#define GAR_STEM_BLOCK_H

#include <geom/BlockDeformer.h>

class StemBlock : public aphid::deform::Block {

	int m_age;
	int m_geomInd;
	float m_exclR;
	
public:
	StemBlock(StemBlock* parent = 0);
	
	void setGeomInd(int x);
	void setExclR(float x);

	const int& age() const;
	const int& geomInd() const;
	const float& exclR() const;
	
	StemBlock* childStem(int i) const;
	
protected:

private:	
};

#endif
