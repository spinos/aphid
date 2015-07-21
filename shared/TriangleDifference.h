/*
 *  TriangleDifference.h
 *  aphid
 *
 *  Created by jian zhang on 7/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ModelDifference.h"
class ATriangleMesh;
class Matrix33F;
class TriangleDifference : public ModelDifference {
public:
	TriangleDifference(ATriangleMesh * target);
	virtual ~TriangleDifference();
	
    void requireQ(AGenericMesh * m);
    void requireQ(const std::vector<unsigned> & v);
	void computeQ(ATriangleMesh * mesh);
	Matrix33F * Q();
protected:
	void computeUndeformedV(ATriangleMesh * mesh);
    Matrix33F * undeformedV();
    unsigned * binded();
	Vector3F getV4(const Vector3F & v1, const Vector3F & v2, const Vector3F & v3) const;
private:
	BaseBuffer * m_V;
    BaseBuffer * m_Q;
    BaseBuffer * m_binded;
};