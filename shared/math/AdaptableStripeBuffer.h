/*
 *  AdaptableStripeBuffer.h
 *  aphid
 *
 *  Created by jian zhang on 1/2/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_ADAPTABLE_STRIPE_BUFFER_H
#define APH_ADAPTABLE_STRIPE_BUFFER_H

#include "Vector3F.h"

namespace aphid {

class AdaptableStripeBuffer {
public:
	AdaptableStripeBuffer();
	virtual ~AdaptableStripeBuffer();
	void create(unsigned maxNumPoint);
	void create(unsigned maxNumStripe, unsigned numCvPerStripe);
	unsigned numStripe() const;
	unsigned numPoints() const;
	char canContain(unsigned x) const;

	void begin();
	char next();
	char end() const;
	
	unsigned * numCvs();
	Vector3F * pos();
	Vector3F * col();
	float * width();
	
	unsigned * currentNumCvs();
	Vector3F * currentPos();
	Vector3F * currentCol();
	float * currentWidth();
	
private:
	void clear();
	void init();
private:
	unsigned * m_numCvs;
	Vector3F * m_pos;
	Vector3F * m_col;
	float * m_width;
	unsigned m_maxNumStripe;
	unsigned m_maxNumCv;
	unsigned m_useNumStripe;
	unsigned m_currentStripe;
	Vector3F * m_curPos;
	Vector3F * m_curCol;
	float * m_curWidth;
};

}
#endif
