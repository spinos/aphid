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
#include <QuickSort.h>

MlSkin::MlSkin() : m_numFeather(0), m_faceCalamusStart(0), m_hasFeatherCreated(false)
{
    m_activeIndices.clear();
}

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

bool MlSkin::createFeather(MlCalamus & ori, const Vector3F & pos, float minDistance)
{
	const unsigned iface = ori.faceIdx();
	if(isPointTooCloseToExisting(pos, iface, minDistance)) return false;
	
	m_calamus.expandBy(1);
	MlCalamus * c = m_calamus.asCalamus();
	*c = ori;
	m_calamus.next();
	
	m_activeIndices.push_back(m_numFeather);
	m_numFeather++;
	
	m_hasFeatherCreated = true;
	return true;
}

void MlSkin::growFeather(const Vector3F & direction)
{
    if(m_activeIndices.size() < 1) return;
	if(direction.length() < 10e-3) return;
	
	const float scale = direction.length();
    
	Vector3F d;
	float rotX;
    for(std::vector<unsigned>::iterator it = m_activeIndices.begin(); it != m_activeIndices.end(); ++it) {
        MlCalamus * c = m_calamus.asCalamus(*it);
		Matrix33F space = tangentFrame(c);
        space.inverse();
		
		d = space.transform(direction);
		rotX = d.angleX();
		c->setRotateX(rotX);
		c->setScale(scale);
    }
}

void MlSkin::finishCreateFeather()
{
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
	
	m_activeIndices.clear();
	m_hasFeatherCreated = false;
}

bool MlSkin::isPointTooCloseToExisting(const Vector3F & pos, const unsigned faceIdx, float minDistance)
{
	resetCollisionRegion(faceIdx);
	
	const unsigned maxCountPerFace = m_numFeather / 2;
	
	Vector3F d, p;
	for(unsigned i=0; i < numRegionElements(); i++) {
		
		unsigned ifeather = m_faceCalamusStart[regionElementIndex(i)];
		for(unsigned j = 0; j < maxCountPerFace; j++) {
			MlCalamus *c = getCalamus(ifeather);
			if(c->faceIdx() != regionElementIndex(i)) break;
			
			getPointOnBody(c, p);
			
			d = p - pos;
			if(d.length() < minDistance) return true;
			
			ifeather++;
		}
	}

	return false;
}

void MlSkin::resetCollisionRegion(unsigned idx)
{
	if(idx == regionElementStart()) return;
	CollisionRegion::resetCollisionRegion(idx);
	m_topo->growAroundQuad(idx, *regionElementIndices());
}

void MlSkin::closestPoint(const Vector3F & origin, IntersectionContext * ctx) const
{
	for(unsigned i=0; i < numRegionElements(); i++) {
		m_body->closestPoint(regionElementIndex(i), origin, ctx);
	}
}

unsigned MlSkin::numFeathers() const
{
	return m_numFeather;
}

MlCalamus * MlSkin::getCalamus(unsigned idx) const
{
	return m_calamus.asCalamus(idx);
}

unsigned MlSkin::numActiveFeather() const
{
	return m_activeIndices.size();
}

MlCalamus * MlSkin::getActive(unsigned idx) const
{
	return m_calamus.asCalamus(m_activeIndices[idx]);
}

void MlSkin::getPointOnBody(MlCalamus * c, Vector3F &p) const
{
	m_body->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
}

Matrix33F MlSkin::tangentFrame(MlCalamus * c) const
{
	return m_body->tangentFrame(c->faceIdx(), c->patchU(), c->patchV());
}

bool MlSkin::hasFeatherCreated() const
{
    return m_hasFeatherCreated;
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
