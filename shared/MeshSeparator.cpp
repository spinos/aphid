/*
 *  MeshSeparator.cpp
 *  aphid
 *
 *  Created by jian zhang on 8/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "MeshSeparator.h"
#include "ATriangleMesh.h"

MeshSeparator::MeshSeparator() 
{
	sdb::TreeNode::MaxNumKeysPerNode = 256;
	sdb::TreeNode::MinNumKeysPerNode = 32;
}

MeshSeparator::~MeshSeparator() {}

void MeshSeparator::separate(ATriangleMesh * m)
{
	clearLastPatch();
	unsigned tdrift = 0;
	m_patchTriangleBegins.clear();
	m_patchTriangleBegins.push_back(tdrift);
	const unsigned nt = m->numTriangles();
	unsigned i=0;
	for(;i<nt;i++) {
		unsigned * tri = m->triangleIndices(i);
		if(!isVerticesConnectedToLastPatch(tri)) {
			clearLastPatch();
			
			//std::cout<<"\n v "<<tri[0]<<","<<tri[1]<<","<<tri[2]
			//<<" drift "<<tdrift;
		
			m_patchTriangleBegins.push_back(tdrift);
			
		}
		connectVerticesToLastPatch(tri);
		tdrift++;
	}
}

bool MeshSeparator::isVerticesConnectedToLastPatch(unsigned * v)
{
// minimal n tri in a patch
	if(m_lastPatch.size() < 9) return true;
	if(m_lastPatch.find(v[0])) return true;
	if(m_lastPatch.find(v[1])) return true;
	if(m_lastPatch.find(v[2])) return true;
	return false;
}

void MeshSeparator::clearLastPatch()
{ m_lastPatch.clear(); }

void MeshSeparator::connectVerticesToLastPatch(unsigned * v)
{
	char * a = new char[1];
	m_lastPatch.insert(v[0], a);
	char * b = new char[1];
	m_lastPatch.insert(v[1], b);
	char * c = new char[1];
	m_lastPatch.insert(v[2], c);
}

unsigned MeshSeparator::numPatches() const
{ return m_patchTriangleBegins.size(); }

unsigned MeshSeparator::patchTriangleBegin(unsigned i) const
{ return m_patchTriangleBegins[i]; }
//:~