/*
 *  DOP8Builder.h
 *  
 *
 *  Created by jian zhang on 1/11/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_MATH_DOP_8_BUILDER_H
#define APH_MATH_DOP_8_BUILDER_H

#include <math/Vector3F.h>

namespace aphid {

class AOrientedBox;

class DOP8Builder {

	Vector3F m_vert[18];
	Vector3F m_facevert[84];
	Vector3F m_facenor[84];
	int m_tri[84];
	int m_ntri;
	
public:
	DOP8Builder();
	void build(const AOrientedBox & ob);
	const int & numTriangles() const;
	const Vector3F * vertex() const;
	const Vector3F * normal() const;
	const int * triangleIndices() const;
	
};

}
#endif