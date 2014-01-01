/*
 *  AdaptableStripeBuffer.h
 *  aphid
 *
 *  Created by jian zhang on 1/2/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <AllMath.h>

class AdaptableStripeBuffer {
public:
	AdaptableStripeBuffer();
	virtual ~AdaptableStripeBuffer();
	void create(unsigned maxNumStripe, unsigned numCvPerStripe);
	unsigned numStripe() const;
	void begin();
	void next();
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
private:
	unsigned * m_numCvs;
	Vector3F * m_pos;
	Vector3F * m_col;
	float * m_width;
	unsigned m_maxNumStripe;
	unsigned m_maxNumCv;
	unsigned m_useNumStripe;
	unsigned m_currentStripe;
};