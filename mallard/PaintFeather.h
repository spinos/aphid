/*
 *  PaintFeather.h
 *  mallard
 *
 *  Created by jian zhang on 1/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
class CalamusSkin;
class FloodCondition;
class PaintFeather {
public:
	enum PaintMode {
		MDirection = 0,
		MLength = 1,
		MWidth = 2,
		MRoll = 3
	};
	
	PaintFeather(CalamusSkin * skin, std::deque<unsigned> * indices, FloodCondition * density);
	virtual ~PaintFeather();
	
	void computeWeights(const Vector3F & center, const float & radius);
	void perform(PaintMode mode, const Vector3F & dv);
protected:

private:
	void brushDirection(const Vector3F & dv);
	void brushLength(const Vector3F & dv);
	void brushWidth(const Vector3F & dv);
	void brushRoll(const Vector3F & dv);
private:
	std::deque<unsigned> * m_indices;
	boost::scoped_array<float> m_weights;
	CalamusSkin * m_skin;
	FloodCondition * m_density;
};