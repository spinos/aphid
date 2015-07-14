/*
 *  TriangleDifference.cpp
 *  aphid
 *
 *  Created by jian zhang on 7/15/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 *  reference: Deformation Transfer for Triangle Meshes by Robert W. Sumner Jovan Popovic
 *  
 */

#include "TriangleDifference.h"
#include "ATriangleMesh.h"
#include "BaseBuffer.h"
#include "Matrix33F.h"
#include <cmath>

TriangleDifference::TriangleDifference(ATriangleMesh * target) : ModelDifference(target)
{
	m_V = new BaseBuffer;
	m_V->create(target->numTriangles() * 36);
	Matrix33F * v = (Matrix33F *)m_V->data();
	computeV(v, target);
	const unsigned n = target->numTriangles();
	unsigned i=0;
	for(;i<n;i++) v[i].inverse();
}

TriangleDifference::~TriangleDifference() 
{
	delete m_V;
}

void TriangleDifference::computeV(Matrix33F * dst, ATriangleMesh * mesh)
{
	const unsigned n = mesh->numTriangles();
	Vector3F * p = mesh->points();
	Vector3F v1, v2, v3, v4;
	unsigned i=0;
	for(;i<n;i++) {
		unsigned * vi = mesh->triangleIndices(i);
		v1 = p[vi[0]];
		v2 = p[vi[1]];
		v3 = p[vi[2]];
		v4 = getV4(v1, v2, v3);
		
		dst[i].fill(v2-v1, v3-v1, v4-v1);
	}
}

Vector3F TriangleDifference::getV4(const Vector3F & v1, const Vector3F & v2, const Vector3F & v3) const
{
	Vector3F nor = (v2 - v1).cross(v3 - v1);
	nor *= 1.f/sqrt(nor.length());
	return v1 + nor;
}

Matrix33F * TriangleDifference::undeformedV()
{ return (Matrix33F *)m_V->data(); }

void TriangleDifference::computeQ(Matrix33F * dst, unsigned n, unsigned * ind, ATriangleMesh * mesh)
{
	Vector3F * p = mesh->points();
	Vector3F v1, v2, v3, v4;
	Matrix33F deformedV;
	unsigned i = 0;
	unsigned idx;
	for(;i<n;i++) {
		idx = ind[i];
		
		unsigned * vi = mesh->triangleIndices(idx);
		v1 = p[vi[0]];
		v2 = p[vi[1]];
		v3 = p[vi[2]];
		v4 = getV4(v1, v2, v3);
		deformedV.fill(v2-v1, v3-v1, v4-v1);
		
		dst[i] = deformedV * undeformedV()[idx];
	}
}
//:~