/*
 *  InverseBilinearInterpolate.h
 *  mallard
 *
 *  Created by jian zhang on 9/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <AllMath.h>
namespace aphid {

class InverseBilinearInterpolate {
public:
	InverseBilinearInterpolate();
	~InverseBilinearInterpolate();
	void setVertices(const Vector3F & a, const Vector3F & b, const Vector3F & c, const Vector3F & d);
	Vector2F operator()(const Vector3F &P);
	
private:
	Vector2F evalBiLinear(const Vector2F& uv) const;
	Vector2F solve(Vector2F M1, Vector2F M2, Vector2F b, bool safeInvert);
	Matrix44F m_space;
	Vector2F m_E, m_F, m_G;
};

}