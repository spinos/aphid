/*
 *  AGenericMesh.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <geom/AGenericMesh.h>
#include <math/BaseBuffer.h>

namespace aphid {

AGenericMesh::AGenericMesh() 
{
	m_points = new BaseBuffer;
	m_normals = new BaseBuffer;
	m_indices = new BaseBuffer;
	m_anchors = new BaseBuffer;
	m_numPoints = m_numIndices = 0;
}

AGenericMesh::~AGenericMesh() 
{
	delete m_points;
	delete m_normals;
	delete m_indices;
	delete m_anchors;
}

const TypedEntity::Type AGenericMesh::type() const
{ return TGenericMesh; }

const BoundingBox AGenericMesh::calculateBBox() const
{
	BoundingBox box;
	const unsigned nv = numPoints();
	Vector3F * p = points();
	for(unsigned i = 0; i < nv; i++) {
		box.updateMin(p[i]);
		box.updateMax(p[i]);
	}
	return box;
}

void AGenericMesh::createBuffer(unsigned np, unsigned ni)
{
	m_points->create(np * 12);
	m_normals->create(np * 12);
	m_indices->create(ni * 4);
	m_anchors->create(np * 4);
}

Vector3F * AGenericMesh::points() const
{ return (Vector3F *)m_points->data(); }

Vector3F * AGenericMesh::vertexNormals() const
{ return (Vector3F *)m_normals->data(); }
	
unsigned * AGenericMesh::indices() const
{ return (unsigned *)m_indices->data(); }

unsigned * AGenericMesh::anchors() const
{ return (unsigned *)m_anchors->data(); }

void AGenericMesh::setNumPoints(unsigned n)
{ m_numPoints = n; }

void AGenericMesh::setNumIndices(unsigned n)
{ m_numIndices = n; }

const unsigned AGenericMesh::numPoints() const
{ return m_numPoints; }

const unsigned AGenericMesh::numIndices() const
{ return m_numIndices; }

void AGenericMesh::clearAnchors()
{ 
	unsigned * anchor = (unsigned *)m_anchors->data();
	unsigned i=0;
	for(; i < m_numPoints; i++) anchor[i] = 0;
}

void AGenericMesh::copyStripe(AGenericMesh * inmesh, unsigned driftP, unsigned driftI)
{
	Vector3F * dstP = & ((Vector3F *)m_points->data())[driftP];
	unsigned * dstA = & ((unsigned *)m_anchors->data())[driftP];
	unsigned * dstI = & ((unsigned *)m_indices->data())[driftI];
	const unsigned np = inmesh->numPoints();
	const unsigned ni = inmesh->numIndices();
	unsigned i=0;
	Vector3F * srcP = inmesh->points();
	unsigned * srcA = inmesh->anchors();
	unsigned * srcI = inmesh->indices();
	for(;i<np;i++) dstP[i] = srcP[i]; 
	for(i=0;i<np;i++) dstA[i] = srcA[i]; 
	for(i=0;i<ni;i++) dstI[i] = driftP + srcI[i]; 
}

const unsigned AGenericMesh::numAnchoredPoints() const
{
	unsigned count = 0;
	unsigned * a = anchors();
	const unsigned n = numPoints();
	unsigned i=0;
	for(;i<n;i++) {
		if(a[i] > 0) count++;
	}
	return count;
}

void AGenericMesh::getAnchorInd(std::map<unsigned, unsigned> & dst) const
{
	unsigned * a = anchors();
	const unsigned n = numPoints();
	unsigned i=0;
	for(;i<n;i++) {
		if(a>0) dst[(a[i]<<8)>>8] = 1;
	}
}

const Vector3F AGenericMesh::averageP() const
{
    Vector3F center = Vector3F::Zero;
    const unsigned n = numPoints();
    Vector3F * p = points();
	unsigned i=0;
	for(;i<n;i++) center += p[i];
    center *= 1.f/(float)n;
	return center;
}

void AGenericMesh::moveIntoSpace(const Matrix44F & m)
{
    const unsigned n = numPoints();
    Vector3F * p = points();
    unsigned i=0;
	for(;i<n;i++) p[i] = m.transform(p[i]);
}

BaseBuffer * AGenericMesh::pointsBuf() const
{ return m_points; }

}
//:~