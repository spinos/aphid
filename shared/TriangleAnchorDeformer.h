/*
 *  TriangleAnchorDeformer.h
 *  aphid
 *
 *  Created by jian zhang on 7/19/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ADeformer.h"
class TriangleDifference;
class ATriangleMesh;
class TriangleAnchorDeformer : public ADeformer {
public:
	TriangleAnchorDeformer();
	virtual ~TriangleAnchorDeformer();
	
	void setDifference(TriangleDifference * diff);
	virtual void setMesh(AGenericMesh * mesh);
	virtual void reset(ATriangleMesh * restM);
	virtual bool solve(ATriangleMesh * m);
protected:
	Vector3F * localP() const;
private:
	TriangleDifference * m_diff;
	BaseBuffer * m_localP;
};