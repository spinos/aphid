/*
 *  AQuadMesh.cpp
 *  
 *
 *  Created by jian zhang on 8/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AQuadMesh.h"
#include "BaseBuffer.h"
#include <iostream>

namespace aphid {

AQuadMesh::AQuadMesh() 
{}

AQuadMesh::~AQuadMesh() 
{}

Vector3F * AQuadMesh::quadP(const int & u, const int & v)
{ return &points()[v*m_nppu + u]; }

const TypedEntity::Type AQuadMesh::type() const
{ return TQuadMesh; }

const unsigned AQuadMesh::numComponents() const
{ return numQuads(); }

const unsigned AQuadMesh::numQuads() const
{ return numIndices() >> 2; }

const BoundingBox AQuadMesh::calculateBBox(unsigned icomponent) const
{
	Vector3F * p = points();
	unsigned * v = quadIndices(icomponent);
	BoundingBox box;
	box.expandBy(p[v[0]]);
	box.expandBy(p[v[1]]);
	box.expandBy(p[v[2]]);
	box.expandBy(p[v[3]]);
	return box;
}

void AQuadMesh::create(const int & useg, const int & vseg)
{
	m_nppu = useg+1;
	unsigned np = m_nppu*(vseg+1);
	unsigned nq = useg*vseg;
	createBuffer(np, nq * 4);
	setNumPoints(np);
	setNumIndices(nq * 4);
	
	int i, j;
	for(j=0;j<vseg;++j) {
		for(i=0;i<useg;++i) {
			unsigned *x = quadIndices(j*useg+i);
			x[0] = j*m_nppu + i;
			x[1] = x[0] + 1;
			x[2] = (j+1)*m_nppu + i + 1;
			x[3] = x[2] - 1;
		}
	}
}

unsigned * AQuadMesh::quadIndices(unsigned idx) const
{ return &indices()[idx<<2]; }

std::string AQuadMesh::verbosestr() const
{
	std::stringstream sst;
	sst<<" quad mesh nv "<<numPoints()
		<<"\n nquad "<<numQuads()
		<<"\n";
	return sst.str();
}

}
//:~
