/*
 *  DrawTriangle.h
 *  
 *
 *  Created by jian zhang on 1/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_Triangle_H
#define APH_OGL_DRAW_Triangle_H

#include <boost/scoped_array.hpp>

namespace aphid {

class Vector3F;

class DrawTriangle {

	boost::scoped_array<Vector3F> m_triNormalBuf;
	boost::scoped_array<Vector3F> m_triColorBuf;
	boost::scoped_array<Vector3F> m_triPositionBuf;
	int m_triBufLength;
	
public:
	DrawTriangle();
	virtual ~DrawTriangle();
	
	const int & triBufLength() const;
	void drawWiredTriangles() const;
	void drawSolidTriangles() const;
	
	const float * triNormalBuf() const;
	const float * triColorBuf() const;
	const float * triPositionBuf() const;

protected:
	Vector3F * triNormalR();
	Vector3F * triPositionR();
	Vector3F * triColorR();
	void setTriDrawBufLen(const int & x);
	void buildTriangleDrawBuf(const int & nt, 
				const int * tri,
				const int & nv, 
				const Vector3F * vertP, 
				const Vector3F * vertN,
				const Vector3F * vertC );
				
};

}
#endif
