/*
 *  DrawPoint.h
 *  
 *
 *  Created by jian zhang on 1/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_POINT_H
#define APH_OGL_DRAW_POINT_H

#include <boost/scoped_array.hpp>

namespace aphid {

class Vector3F;

class DrawPoint {

	boost::scoped_array<Vector3F> m_pntNormalBuf;
	boost::scoped_array<Vector3F> m_pntPositionBuf;
	boost::scoped_array<Vector3F> m_pntColorBuf;
	int m_pntBufLength;
	
public:
	DrawPoint();
	virtual ~DrawPoint();
	
	const int & pntBufLength() const;
	void drawPoints() const;
	void drawWiredPoints() const;
	
	const float * pntNormalBuf() const;
	const float * pntPositionBuf() const;
	const float * pntColorBuf() const;
	
	Vector3F * pntNormalR();
	Vector3F * pntPositionR();
	Vector3F * pntColorR();
	void setPointDrawBufLen(const int & x);
	
protected:
	void buildPointDrawBuf(const int & nv,
				const float * vertP, 
				const float * vertN,
				const float * vertC,
				int stride = 0);
				
};

}
#endif
