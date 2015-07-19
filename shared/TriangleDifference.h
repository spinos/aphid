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
	void computeQ(ATriangleMesh * mesh);
protected:
	void computeV(Matrix33F * dst, ATriangleMesh * mesh) const;
    void computeC(Vector3F * dst, ATriangleMesh * mesh) const;
	Matrix33F * undeformedV();
    Matrix33F * Q();
    unsigned * binded();
    Vector3F * undeformedC();
	Vector3F getV4(const Vector3F & v1, const Vector3F & v2, const Vector3F & v3) const;
private:
	BaseBuffer * m_V;
    BaseBuffer * m_Q;
    BaseBuffer * m_binded;
    BaseBuffer * m_C;
};