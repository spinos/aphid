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
	clearPatches();
	
	unsigned tdrift = 0;
	const unsigned nt = m->numTriangles();
	unsigned i=0;
	int iPatch=-1;
	for(;i<nt;i++) {
		unsigned * tri = m->triangleIndices(i);
		if(!isVerticesConnectedToAnyPatch(tri, iPatch)) {
			addPatch(tdrift);
			iPatch = tdrift;
		}
		connectVerticesToPatch(tri, iPatch);
		tdrift++;
	}
	mergeConnectedPatches();
}

void MeshSeparator::clearPatches()
{
	std::map<unsigned, VertexIndices *>::iterator it = m_patches.begin();
	for(;it!=m_patches.end();++it) delete it->second;
	m_patches.clear();
}

unsigned MeshSeparator::numPatches() const
{ return m_patches.size(); }

bool MeshSeparator::isVerticesConnectedToAnyPatch(unsigned * v, int & ipatch)
{
	if(m_patches.size() < 1) {
		ipatch = -1;
		return false;
	}
	
	std::map<unsigned, VertexIndices *>::iterator it = m_patches.begin();
	for(;it!=m_patches.end();++it) {
		ipatch = it->first;
		VertexIndices * p = it->second;
		if(p->find(v[0])) return true;
		if(p->find(v[1])) return true;
		if(p->find(v[2])) return true;
		
	}
	ipatch = -1;
	return false;
}

void MeshSeparator::addPatch(unsigned x)
{
	VertexIndices * p = new VertexIndices;
	m_patches[x] = p;
}

void MeshSeparator::connectVerticesToPatch(unsigned * v, unsigned ipatch)
{
	VertexIndices * p = m_patches[ipatch];
	
	char * a = new char[1];
	p->insert(v[0], a);
	char * b = new char[1];
	p->insert(v[1], b);
	char * c = new char[1];
	p->insert(v[2], c);
}

void MeshSeparator::mergeConnectedPatches()
{
	std::map<unsigned, VertexIndices *>::iterator it = m_patches.begin();
	unsigned connected;
	it++;
	for(;it!=m_patches.end();++it) {
		if(isPatchConnectedToAnyPatch(it->first, connected, it->second)) {
			std::cout<<"\n patch"<<it->first<<" is connected to patch"<<connected;
			mergePatches(connected, it->first);
			removePatch(it->first);
			it--;
		}
	}
}

bool MeshSeparator::isPatchConnectedToAnyPatch(unsigned t, unsigned & connectedPatch, VertexIndices * v)
{
	std::map<unsigned, VertexIndices *>::iterator it = m_patches.begin();
	for(;it!=m_patches.end();++it) {
		if(it->first < t) {
			if(it->second->intersect(v)) {
				connectedPatch = it->first;
				return true;
			}
		}
	}
	return false;
}

void MeshSeparator::mergePatches(unsigned a, unsigned b)
{
	VertexIndices * dst = m_patches[a];
	VertexIndices * src = m_patches[b];
	src->begin();
	while(!src->end()) {
		char * v = new char[1];
		dst->insert(src->key(), v);
		src->next();
	}
}

void MeshSeparator::removePatch(unsigned a)
{
	std::map<unsigned, VertexIndices *>::iterator it = m_patches.find(a);
	delete it->second;
	m_patches.erase(it);
}
//:~