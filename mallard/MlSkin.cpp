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
	resetFaceCalamusIndirection();
}

AccPatchMesh * MlSkin::bodyMesh() const
{
	return m_body;
}

void MlSkin::floodAround(MlCalamus floodC, unsigned floodFaceIdx, const Vector3F & floodPos, const Vector3F & floodNor, const float & floodMaxD, const float & floodMinD)
{

	resetCollisionRegionAround(floodFaceIdx, floodPos, floodMaxD);
	std::vector<unsigned> growOnFaces;
	unsigned i, j, iface;
	
	for(i = 0; i < numRegionElements(); i++)
		growOnFaces.push_back(regionElementIndex(i));
	
	float u, v;
	Vector3F adart, facing;
	std::vector<Vector3F> darts;
	for(i = 0; i < growOnFaces.size(); i++) {
		iface = growOnFaces[i];
		const unsigned ndart = 4 + m_body->calculateBBox(iface).area() / floodMinD / floodMinD;
		for(j = 0; j < ndart; j++) {
		
			u = rand()%91/91.f;
			v = rand()%97/97.f;
			m_body->pointOnPatch(iface, u, v, adart);
			
			if(Vector3F(floodPos, adart).length() > floodMaxD) continue;
			
			m_body->normalOnPatch(iface, u, v, facing);
			
			if(facing.dot(floodNor) < .67f) continue;
			
			if(isPointTooCloseToExisting(adart, iface, floodMinD)) continue;
			
			if(!isDartCloseToExisting(adart, darts, floodMinD)) {
				darts.push_back(adart);
				floodC.bindToFace(iface, u, v);
				createFeather(floodC);
			}
		}
	}
	growOnFaces.clear();
	darts.clear();
}

void MlSkin::selectAround(unsigned idx, const Vector3F & pos, const Vector3F & nor, const float & maxD)
{
	resetCollisionRegionAround(idx, pos, maxD);
	const unsigned maxCountPerFace = m_numFeather / 2;
	
	Vector3F d, p, n;
	for(unsigned i=0; i < numRegionElements(); i++) {
		
		unsigned ifeather = m_faceCalamusStart[regionElementIndex(i)];
		for(unsigned j = 0; j < maxCountPerFace; j++) {
			MlCalamus *c = getCalamus(ifeather);
			if(c->faceIdx() != regionElementIndex(i)) break;
			
			getNormalOnBody(c, n);
			if(n.dot(nor) < .67f) continue;
			
			getPointOnBody(c, p);
			
			d = p - pos;
			if(d.length() < maxD) {
				if(!IsElementIn(ifeather, m_activeIndices))
					m_activeIndices.push_back(ifeather);
			}
			
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
	
	m_activeIndices.push_back(m_numFeather);
	
	m_numFeather++;
	m_numCreatedFeather++;
	
	return true;
}

void MlSkin::growFeather(const Vector3F & direction)
{
	//if(!hasFeatherCreated()) return;
	const unsigned num = numActive();
	if(num < 1) return;

	const float scale = direction.length();
    if(scale < 10e-3) return;
	
	Vector3F d;
	float rotX;
	for(unsigned i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		Matrix33F space = tangentSpace(c);
        space.inverse();
		
		d = space.transform(direction);
		rotX = d.angleX();
		c->setRotateX(rotX);
		c->setScale(scale);
    }
}

void MlSkin::combFeather(const Vector3F & direction, const Vector3F & center, const float & radius)
{
	if(direction.length() < 10e-3) return;
	const unsigned num = numActive();
	if(num < 1) return;
	
	Vector3F p;
	float rotX, drop;
	for(unsigned i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		getPointOnBody(c, p);
		drop = Vector3F(p, center).length() / radius;
		drop = 1.f - drop * drop;
		
		Matrix33F space = tangentSpace(c);
		
		Vector3F zdir(0.f, 0.f, c->realScale());
		zdir = rotationFrame(c).transform(zdir);
		zdir += direction * drop * .1f * c->realScale();
		
		space.inverse();
		
		zdir = space.transform(zdir);
		rotX = zdir.angleX();

		c->setRotateX(rotX);
    }
}

void MlSkin::scaleFeather(const Vector3F & direction, const Vector3F & center, const float & radius)
{
	if(direction.length() < 10e-3) return;
	const unsigned num = numActive();
	if(num < 1) return;
	
	Vector3F p;
	float drop;
	unsigned i;
	
	float activeMeanScale = 0.f;
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		activeMeanScale += c->realScale();
	}
	activeMeanScale /= num;
	
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		getPointOnBody(c, p);
		drop = Vector3F(p, center).length() / radius;
		drop = 1.f - drop * drop;
		
		Matrix33F space = rotationFrame(c);
		Vector3F zdir(0.f, 0.f, 1.f);
		zdir = space.transform(zdir);

		float fac = 0.1f;
		if(direction.dot(zdir) < 0.f) fac = -0.1f;

		c->setScale((fac + 1.f) * activeMeanScale * drop + c->realScale() * (1.f - drop));
    }
}

void MlSkin::pitchFeather(const Vector3F & direction, const Vector3F & center, const float & radius)
{
	if(direction.length() < 10e-3) return;
	const unsigned num = numActive();
	if(num < 1) return;
	
	Vector3F p;
	float drop;
	unsigned i;
	
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		getPointOnBody(c, p);
		drop = Vector3F(p, center).length() / radius;
		drop = 1.f - drop * drop;
		
		Matrix33F space = rotationFrame(c);
		Vector3F zdir(0.f, 0.f, 1.f);
		zdir = space.transform(zdir);

		float fac = 0.1f;
		if(direction.dot(zdir) < 0.f) fac = -0.1f;

		c->setRotateY(((fac + 1.f) * drop + (1.f - drop)) * c->rotateY());
    }
}

void MlSkin::finishCreateFeather()
{
    computeFaceCalamusIndirection();
	
	m_numCreatedFeather = 0;
}

void MlSkin::finishEraseFeather()
{
	if(numActive() == numFeathers()) {
		m_numFeather = 0;
		m_calamus->setIndex(0);
		resetFaceCalamusIndirection();
	}
	
	QuickSort::Sort(m_activeIndices, 0, numActive() - 1);
	
	unsigned i, j;
	const unsigned num = numActive();
	for(i = 0; i < num; i++) {
		j = lastInactive();
		if(m_activeIndices[i] < j) {
			m_calamus->swapElement(m_activeIndices[i], j);
			m_activeIndices.push_back(j);
		}
		m_numFeather--;
	}
	
	m_calamus->setIndex(m_numFeather);
	computeFaceCalamusIndirection();
}

void MlSkin::computeFaceCalamusIndirection()
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
}

void MlSkin::resetFaceCalamusIndirection()
{
	for(unsigned i = 0; i < m_body->getNumFaces(); i++) m_faceCalamusStart[i] = 0;
}

unsigned MlSkin::lastInactive() const
{	
	unsigned i = 0;
	for(i = numFeathers() - 1; i > 0; i--) {
		if(!IsElementIn(i, m_activeIndices))
			return i;
	}
	return i;
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

unsigned MlSkin::numActive() const
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

void MlSkin::getNormalOnBody(MlCalamus * c, Vector3F &p) const
{
	m_body->normalOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
}

Matrix33F MlSkin::tangentSpace(MlCalamus * c) const
{
	return m_body->tangentFrame(c->faceIdx(), c->patchU(), c->patchV());
}

Matrix33F MlSkin::rotationFrame(MlCalamus * c) const
{
	Matrix33F frm;
	frm.rotateX(c->rotateX());
	frm.multiply(m_body->tangentFrame(c->faceIdx(), c->patchU(), c->patchV()));
	return frm;
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

void MlSkin::run()
{

}
