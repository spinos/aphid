/*
 *  ATetrahedronMesh.cpp
 *  aphid
 *
 *  Created by jian zhang on 4/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ATetrahedronMesh.h"
#include "BaseBuffer.h"
#include "tetrahedron_math.h"

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

std::string ATetrahedronMesh::verbosestr() const
{
	std::stringstream sst;
	sst<<" tetrahedron mesh nv "<<numPoints()
		<<"\n ntetra "<<numTetrahedrons()
		<<"\n volume "<<volume()
		<<"\n";
	return sst.str();
}
//:~
