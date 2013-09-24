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
#include "MlCalamusArray.h"

MlSkin::MlSkin() : m_numFeather(0), m_faceCalamusStart(0), m_numCreatedFeather(0)
{
    m_activeIndices.clear();
	m_calamus = new MlCalamusArray; 
}

MlSkin::~MlSkin()
{
	m_calamus->clear();
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

void MlSkin::floodAround(MlCalamus c, unsigned idx, const Vector3F & pos, const float & maxD, const float & minD)
{
	resetCollisionRegionAround(idx, pos, maxD);
	std::vector<unsigned> growOnFaces;
	unsigned i, j, iface;
	
	for(i = 0; i < numRegionElements(); i++)
		growOnFaces.push_back(regionElementIndex(i));
	
	float u, v;
	Vector3F adart;
	std::vector<Vector3F> darts;
	for(i = 0; i < growOnFaces.size(); i++) {
		iface = growOnFaces[i];
		const unsigned ndart = 4 + m_body->calculateBBox(iface).area() / minD / minD;
		for(j = 0; j < ndart; j++) {
		
			u = rand()%91/91.f;
			v = rand()%97/97.f;
			m_body->pointOnPatch(iface, u, v, adart);
			
			if(Vector3F(pos, adart).length() > maxD) continue;
			if(isPointTooCloseToExisting(adart, iface, minD)) continue;
			
			if(!isDartCloseToExisting(adart, darts, minD)) {
				darts.push_back(adart);
				c.bindToFace(iface, u, v);
				createFeather(c);
			}
		}
	}
	growOnFaces.clear();
	darts.clear();
}

void MlSkin::selectAround(unsigned idx, const Vector3F & pos, const float & maxD)
{
	resetCollisionRegionAround(idx, pos, maxD);
	const unsigned maxCountPerFace = m_numFeather / 2;
	
	Vector3F d, p;
	for(unsigned i=0; i < numRegionElements(); i++) {
		
		unsigned ifeather = m_faceCalamusStart[regionElementIndex(i)];
		for(unsigned j = 0; j < maxCountPerFace; j++) {
			MlCalamus *c = getCalamus(ifeather);
			if(c->faceIdx() != regionElementIndex(i)) break;
			
			getPointOnBody(c, p);
			
			d = p - pos;
			if(d.length() < maxD) m_activeIndices.push_back(ifeather);
			
			ifeather++;
		}
	}
}

void MlSkin::discardActive()
{
	m_activeIndices.clear();
}

bool MlSkin::createFeather(MlCalamus & ori)
{
	m_calamus->expandBy(1);
	MlCalamus * c = m_calamus->asCalamus();
	*c = ori;
	m_calamus->next();
	
	m_numFeather++;
	m_numCreatedFeather++;
	
	return true;
}

void MlSkin::growFeather(const Vector3F & direction)
{
	if(!hasFeatherCreated()) return;
	const unsigned num = numCreated();
	if(direction.length() < 10e-3) return;
	
	const float scale = direction.length();
    
	Vector3F d;
	float rotX;
	for(unsigned i =0; i < num; i++) {
    MlCalamus * c = getCreated(i);
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
		QuickSort::Sort(*m_calamus, 0, m_numFeather - 1);
		
	unsigned cur;
	unsigned pre = m_body->getNumFaces();
	for(unsigned i = 0; i < m_numFeather; i++) {
		cur = getCalamus(i)->faceIdx();
		if(cur != pre) {
			m_faceCalamusStart[cur] = i;
			pre = cur;
		}
	}
	
	m_numCreatedFeather = 0;
}

void MlSkin::finishEraseFeather()
{

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

bool MlSkin::isDartCloseToExisting(const Vector3F & pos, const std::vector<Vector3F> & existing, float minDistance) const
{
	std::vector<Vector3F>::const_iterator it;
	for(it = existing.begin(); it != existing.end(); ++it) {
		if(Vector3F(pos, *it).length() < minDistance) return true;
	}
	return false;
}

void MlSkin::resetCollisionRegion(unsigned idx)
{
	if(idx == regionElementStart()) return;
	CollisionRegion::resetCollisionRegion(idx);
	m_topo->growAroundQuad(idx, *regionElementIndices());
}

void MlSkin::resetCollisionRegionAround(unsigned idx, const Vector3F & p, const float & d)
{
	CollisionRegion::resetCollisionRegion(idx);
	m_topo->growAroundQuad(idx, *regionElementIndices());
	for(unsigned i = 1; i < numRegionElements(); i++) {
		BoundingBox bb = m_body->calculateBBox(regionElementIndex(i));
		if(bb.isPointAround(p, d))
			m_topo->growAroundQuad(regionElementIndex(i), *regionElementIndices());
	}
	for(unsigned i = 1; i < numRegionElements(); i++) {
		BoundingBox bb = m_body->calculateBBox(regionElementIndex(i));
		if(!bb.isPointAround(p, d)) {
			regionElementIndices()->erase(regionElementIndices()->begin() + i);
			i--;
		}
	}
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
	return m_calamus->asCalamus(idx);
}

unsigned MlSkin::numActiveFeather() const
{
	return m_activeIndices.size();
}

MlCalamus * MlSkin::getActive(unsigned idx) const
{
	return m_calamus->asCalamus(m_activeIndices[idx]);
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
    return m_numCreatedFeather > 0;
}

unsigned MlSkin::numCreated() const
{
	return m_numCreatedFeather;
}

MlCalamus * MlSkin::getCreated(unsigned idx) const
{
	return m_calamus->asCalamus(m_numFeather - m_numCreatedFeather + idx);
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
