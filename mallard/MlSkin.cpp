/*
 *  MlSkin.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlSkin.h"
#include <AccPatchMesh.h>
#include <MeshTopology.h>
#include <iostream>
#include <QuickSort.h>

MlSkin::MlSkin() : m_numFeather(0), m_faceCalamusStart(0) {}
MlSkin::~MlSkin()
{
	m_calamus.clear();
	if(m_faceCalamusStart) delete[] m_faceCalamusStart;
}

void MlSkin::setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo)
{
	m_body = mesh;
	m_topo = topo;
	m_faceCalamusStart = new unsigned[mesh->getNumFaces()];
	for(unsigned i = 0; i < m_body->getNumFaces(); i++) m_faceCalamusStart[i] = 0;
}

AccPatchMesh * MlSkin::bodyMesh() const
{
	return m_body;
}

bool MlSkin::addCalamus(MlCalamus & ori, const Vector3F & pos, float minDistance)
{
	const unsigned iface = ori.faceIdx();
	if(isPointTooCloseToExisting(pos, iface, minDistance)) return false;
	
	m_calamus.expandBy(1);
	MlCalamus * c = m_calamus.asCalamus();
	*c = ori;
	m_calamus.next();
	m_numFeather++;
	
	if(m_numFeather > 1)
		QuickSort::Sort(m_calamus, 0, m_numFeather - 1);
		
	unsigned cur;
	unsigned pre = m_body->getNumFaces();
	for(unsigned i = 0; i < m_numFeather; i++) {
		cur = getCalamus(i)->faceIdx();
		if(cur != pre) {
			m_faceCalamusStart[cur] = i;
			pre = cur;
		}
	}
	
	return true;
}

bool MlSkin::isPointTooCloseToExisting(const Vector3F & pos, const unsigned faceIdx, float minDistance) const
{
	std::vector<unsigned> conn;
	m_topo->growAroundQuad(faceIdx, conn);
	
	const unsigned maxCountPerFace = m_numFeather / 2;
	
	Vector3F d, p;
	for(unsigned i=0; i < conn.size(); i++) {
		
		unsigned ifeather = m_faceCalamusStart[conn[i]];
		for(unsigned j = 0; j < maxCountPerFace; j++) {
			MlCalamus *c = getCalamus(ifeather);
			if(c->faceIdx() != conn[i]) break;
			
			m_body->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
			
			d = p - pos;
			if(d.length() < minDistance) return true;
			
			ifeather++;
		}
	}

	return false;
}

unsigned MlSkin::numFeathers() const
{
	return m_numFeather;
}

MlCalamus * MlSkin::getCalamus(unsigned idx) const
{
	return m_calamus.asCalamus(idx);
}

void MlSkin::verbose() const
{
	std::cout<<"face id\n";
	for(unsigned i = 0; i < m_numFeather; i++) {
		std::cout<<" "<<getCalamus(i)->faceIdx();
	}
	std::cout<<"\n";
	
	std::cout<<"face start\n";
	for(unsigned i = 0; i < m_body->getNumFaces(); i++) {
		if(m_faceCalamusStart[i] > 0) std::cout<<" "<<i<<":"<<m_faceCalamusStart[i];
	}
	std::cout<<"\n";
}
