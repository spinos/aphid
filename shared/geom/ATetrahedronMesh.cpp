/*
 *  ATetrahedronMesh.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <geom/ATetrahedronMesh.h>
#include <foundation/BaseBuffer.h>
#include "tetrahedron_math.h"
#include <GjkIntersection.h>
#include <geom/ClosestToPointTest.h>

namespace aphid {

ATetrahedronMesh::ATetrahedronMesh() 
{
}

ATetrahedronMesh::~ATetrahedronMesh() 
{
}

const TypedEntity::Type ATetrahedronMesh::type() const
{ return TTetrahedronMesh; }

const unsigned ATetrahedronMesh::numComponents() const
{ return numTetrahedrons(); }

const unsigned ATetrahedronMesh::numTetrahedrons() const
{ return numIndices() / 4; }

const BoundingBox ATetrahedronMesh::calculateBBox(unsigned icomponent) const
{
	Vector3F * p = points();
	unsigned * v = &indices()[icomponent*4];
	BoundingBox box;
	box.updateMin(p[v[0]]);
	box.updateMax(p[v[0]]);
	box.updateMin(p[v[1]]);
	box.updateMax(p[v[1]]);
	box.updateMin(p[v[2]]);
	box.updateMax(p[v[2]]);
	box.updateMin(p[v[3]]);
	box.updateMax(p[v[3]]);
	return box;
}

bool ATetrahedronMesh::intersectBox(unsigned icomponent, const BoundingBox & box)
{
    BoundingBox tbox = calculateBBox(icomponent);
    if(!tbox.intersect(box)) return false;
    
    Vector3F * p = points();
	unsigned * v = tetrahedronIndices(icomponent);
    return gjk::IntersectTest::evaluateTetrahedron(p, v);
}

void ATetrahedronMesh::create(unsigned np, unsigned nt)
{
	createBuffer(np, nt * 4);
	setNumPoints(np);
	setNumIndices(nt * 4);
}

unsigned * ATetrahedronMesh::tetrahedronIndices(unsigned idx) const
{
	return &indices()[idx*4];
}

float ATetrahedronMesh::calculateVolume() const
{
    Vector3F * p = points();
	unsigned * v = indices();
    
    const unsigned n = numTetrahedrons();
    float sum = 0.f;
    Vector3F q[4];
    unsigned i = 0;
    for(;i<n;i++) {
        q[0] = p[v[0]];
		q[1] = p[v[1]];
		q[2] = p[v[2]];
		q[3] = p[v[3]];
        sum+=tetrahedronVolume(q);
        v+=4;
    }
    
    return sum;
}

const float ATetrahedronMesh::volume() const
{ return m_volume; }

void ATetrahedronMesh::setVolume(float x)
{ m_volume = x;}

void ATetrahedronMesh::closestToPoint(unsigned icomponent, ClosestToPointTestResult * result)
{
	if(result->_distance < 0.f) return;
	Vector3F * p = points();
	unsigned * v = tetrahedronIndices(icomponent);
    
	Vector3F q[4];
	q[0] = p[v[0]];
	q[1] = p[v[1]];
	q[2] = p[v[2]];
	q[3] = p[v[3]];
	
	Float4 coord;
	if(pointInsideTetrahedronTest(result->_toPoint, q)) {
		result->_distance = -1.f;
		result->_hasResult = true;
		result->_icomponent = icomponent;
		result->_isInside = true;
		
		coord = getBarycentricCoordinate4(result->_toPoint, q);
		result->_contributes[0] = coord.x;
		result->_contributes[1] = coord.y;
		result->_contributes[2] = coord.z;
		result->_contributes[3] = coord.w;
		return;
	}
	
	Vector3F clamped = closestPOnTetrahedron(q, result->_toPoint);
	float d = clamped.distanceTo(result->_toPoint);
	
	if(d>=result->_distance) return;
	
	result->_distance = d;
	result->_hasResult = true;
	result->_icomponent = icomponent;
	result->_isInside = false;
	
	coord = getBarycentricCoordinate4(clamped, q);
	result->_contributes[0] = coord.x;
	result->_contributes[1] = coord.y;
	result->_contributes[2] = coord.z;
	result->_contributes[3] = coord.w;
}

TetrahedronSampler * ATetrahedronMesh::sampler()
{ return &m_sampler; }

void ATetrahedronMesh::checkTetrahedronVolume()
{
	const unsigned n = numTetrahedrons();
	Vector3F * pos = points();
	Vector3F p[4];
	unsigned tmp;
	unsigned i = 0;
	for(;i<n;i++) {
		unsigned * tet = tetrahedronIndices(i);
		p[0] = pos[tet[0]];
		p[1] = pos[tet[1]];
		p[2] = pos[tet[2]];
		p[3] = pos[tet[3]];
		
		if(tetrahedronVolume(p)<0.f) {
			tmp = tet[1];
			tet[1] = tet[2];
			tet[2] = tmp;
#if 0			
			p[0] = pos[tet[0]];
			p[1] = pos[tet[1]];
			p[2] = pos[tet[2]];
			p[3] = pos[tet[3]];
			
			std::cout<<" tet vol after swap 1 2 "<<tetrahedronVolume(p)<<"\n";
#endif
		}
	}
}

std::string ATetrahedronMesh::verbosestr() const
{
	std::stringstream sst;
	sst<<" tetrahedron mesh nv "<<numPoints()
		<<"\n ntetra "<<numTetrahedrons()
		<<"\n volume "<<volume()
		<<"\n";
	return sst.str();
}

}
//:~