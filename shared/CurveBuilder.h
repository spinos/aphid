/*
 *  CurveBuilder.h
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "BaseCurve.h"
#include <vector>
class CurveBuilder {
public:
	CurveBuilder();
	virtual ~CurveBuilder();
	void addVertex(const Vector3F & vert);
	void finishBuild(BaseCurve * c);
private:
	static std::vector<Vector3F> BuilderVertices;
	
};