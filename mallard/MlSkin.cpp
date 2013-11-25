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
#include <BaseImage.h>
#include <SelectCondition.h>
#include <FloodCondition.h>
#include "MlCalamusArray.h"

MlSkin::MlSkin() : m_numFeather(0), m_faceCalamusTable(0), m_numCreatedFeather(0)
{
    m_activeIndices.clear();
	m_calamus = new MlCalamusArray;
}

MlSkin::~MlSkin()
{
	cleanup();
}

void MlSkin::cleanup()
{
	clearFeather();
	if(m_faceCalamusTable) {
		delete[] m_faceCalamusTable;
		m_faceCalamusTable = 0;
	}
	
}

void MlSkin::clearFeather()
{
	m_calamus->clear();
	m_calamus->initialize();
	m_numFeather = 0;
}

void MlSkin::setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo)
{
	CollisionRegion::setBodyMesh(mesh, topo);
	m_faceCalamusTable = new FloodTable[mesh->getNumFaces()];
	resetFaceCalamusIndirection();
}

void MlSkin::floodAround(MlCalamus floodC, FloodCondition * condition)
{	
	unsigned i, j, iface;
	
	float u, v;
	Vector3F adart, facing;
	std::vector<Vector3F> darts;
	for(i = 0; i < m_floodFaces.size(); i++) {
		iface = m_floodFaces[i].faceIdx;
		m_floodFaces[i].dartBegin = m_floodFaces[i].dartEnd = darts.size();
		const unsigned ndart = 4 + bodyMesh()->calculateBBox(iface).area() / condition->minDistance() / condition->minDistance();
		for(j = 0; j < ndart; j++) {
			if(condition->filteredByProbability()) continue;
		
			u = ((float)(rand()%591))/591.f;
			v = ((float)(rand()%593))/593.f;
			bodyMesh()->pointOnPatch(iface, u, v, adart);
			
			if(condition->byDistance()) {
				if(condition->filteredByDistance(adart)) continue;
			}
			
			if(hasActiveRegion() && condition->byRegion()) {
				if(!sampleColorMatches(iface, u, v)) continue;
			}
			
			//bodyMesh()->normalOnPatch(iface, u, v, facing);
			//if(facing.dot(floodNor) < .23f) continue;
			resetCollisionRegion(iface);
	
			if(isPointTooCloseToExisting(adart, condition->minDistance())) continue;
			
			if(isDartCloseToExisting(adart, darts, condition->minDistance())) continue;
				
			darts.push_back(adart);
			floodC.bindToFace(iface, u, v);
			createFeather(floodC);
			m_floodFaces[i].dartEnd++;
		}
	}
	darts.clear();
}

void MlSkin::selectAround(unsigned idx, SelectCondition * selcon)
{	
	resetCollisionRegionAround(idx, selcon->center(), selcon->maxDistance());
	
	for(unsigned i=0; i < numRegionElements(); i++) {
		unsigned preCount = m_activeIndices.size();
		if(selectFeatherByFace(regionElementIndex(i), selcon) > 0) {
			FloodTable t(regionElementIndex(i));
			t.dartBegin = preCount;
			t.dartEnd = m_activeIndices.size();
			m_activeFaces.push_back(t);
		}
	}
}

unsigned MlSkin::selectFeatherByFace(unsigned faceIdx, SelectCondition * selcon)
{
	Vector3F p, n;
	const unsigned preCount = m_activeIndices.size();
	const unsigned featherBegin = m_faceCalamusTable[faceIdx].dartBegin;
	const unsigned featherEnd = m_faceCalamusTable[faceIdx].dartEnd;
	for(unsigned j = featherBegin; j < featherEnd; j++) {
		MlCalamus *c = getCalamus(j);
		if(c->faceIdx() != faceIdx) break;
		
		if(selcon->filteredByProbability()) continue;
		
		if(selcon->byFacing()) {
			getNormalOnBody(c, n);
			if(selcon->filteredByFacing(n)) continue;
		}
		
		if(selcon->byDistance()) {
			getPointOnBody(c, p);
			if(selcon->filteredByDistance(p)) continue;
		}
			
		if(hasActiveRegion() && selcon->byRegion()) {
			if(!sampleColorMatches(c->faceIdx(), c->patchU(), c->patchV())) continue;
		}
		
		if(!isActiveFeather(j))
			m_activeIndices.push_back(j);
	}
	return m_activeIndices.size() - preCount;
}

void MlSkin::discardActive()
{
	m_activeIndices.clear();
	m_activeFaces.clear();
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

void MlSkin::setNumFeathers(unsigned num)
{
	m_calamus->expandBy(num);
	m_numFeather = num;
}

void MlSkin::growFeather(const Vector3F & direction)
{
	const unsigned num = numActive();
	if(num < 1) return;

	const float scale = direction.length();
    if(scale < 10e-3) return;
	
	Vector3F d;
	Matrix33F space;
	float rotX;
	for(unsigned i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		tangentSpace(c, space);
        space.inverse();
		
		d = space.transform(direction);
		rotX = d.angleX();
		c->setRotateX(rotX);
		c->setScale(scale);
    }
}

void MlSkin::combFeather(const Vector3F & direction, const Vector3F & center, const float & radius)
{
	if(direction.length() < 10e-4) return;
	const unsigned num = numActive();
	if(num < 1) return;
	
	Matrix33F space, rotfrm;
	Vector3F p, div, zdir;
	float rotX, drop;
	unsigned i;
	
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		getPointOnBody(c, p);
		drop = Vector3F(p, center).length() / radius;
		drop = 1.f - drop * drop;
		
		tangentSpace(c, space);		
		space.inverse();
		
		div = space.transform(direction);
		div.x = 0.f;
		
		zdir.set(0.f, 0.f, 1.f);
		zdir.rotateAroundAxis(Vector3F::XAxis, c->rotateX());
		zdir += div * drop * .5f;
		
		rotX = zdir.angleX();

		c->setRotateX(rotX);
    }
}

void MlSkin::scaleFeather(const Vector3F & direction, const Vector3F & center, const float & radius)
{
	if(direction.length() < 10e-4) return;
	const unsigned num = numActive();
	if(num < 1) return;
	
	Matrix33F space;
	Vector3F p, zdir;
	float drop;
	unsigned i;
	
	float activeMeanScale = 0.f;
	Vector3F activeMeanDir;
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		tangentSpace(c, space);
		
		zdir.set(0.f, 0.f, 1.f);
		zdir.rotateAroundAxis(Vector3F::XAxis, c->rotateX());
		zdir = space.transform(zdir);
		activeMeanDir += zdir;
		activeMeanScale += c->realScale();
	}
	activeMeanScale /= num;
	activeMeanDir /= (float)num;

	if(direction.dot(activeMeanDir) < 0.f) activeMeanScale *= .9f;
	else activeMeanScale *= 1.1f;
	
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		getPointOnBody(c, p);
		drop = Vector3F(p, center).length() / radius;
		drop = 1.f - drop * drop;

		c->setScale(activeMeanScale * drop + c->realScale() * (1.f - drop));
    }
}

void MlSkin::pitchFeather(const Vector3F & direction, const Vector3F & center, const float & radius)
{
	if(direction.length() < 10e-4) return;
	const unsigned num = numActive();
	if(num < 1) return;
	
	Matrix33F space;
	Vector3F p, zdir;
	float drop;
	unsigned i;
	
	float activeMeanPitch = 0.f;
	Vector3F activeMeanDir;
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		activeMeanPitch += c->rotateY();
		
		tangentSpace(c, space);
		zdir.set(0.f, 0.f, 1.f);
		zdir.rotateAroundAxis(Vector3F::XAxis, c->rotateX());
		zdir = space.transform(zdir);
		activeMeanDir += zdir;
	}
	activeMeanPitch /= num;
	activeMeanDir /= (float)num;

	if(direction.dot(activeMeanDir) < 0.f) activeMeanPitch -= .05f;
	else activeMeanPitch += .05f;
	
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		getPointOnBody(c, p);
		drop = Vector3F(p, center).length() / radius;
		drop = 1.f - drop * drop;

		c->setRotateY(activeMeanPitch * drop + c->rotateY() * (1.f - drop));
    }
}

void MlSkin::finishCreateFeather()
{
	if(!hasFeatherCreated()) return;
	if(!bodyMesh()) return;
    computeFaceCalamusIndirection();
	m_numCreatedFeather = 0;
	std::cout<<" increase to "<<numFeathers();
}

void MlSkin::finishEraseFeather()
{
	if(numActive() < 1) return;
	resetFaceCalamusIndirection();
		
	if(numActive() == numFeathers()) {
		m_numFeather = 0;
		m_calamus->setIndex(0);
		std::cout<<" reduce to 0";
		return;
	}
	
	unsigned i, j = m_numFeather - 1;
	m_numFeather -= numActive();
	
	const unsigned num = numActive();
	for(i = 0; i < num; i++) {
		if(m_activeIndices[i] >= m_numFeather) continue;
		j = lastInactive(j);
		if(m_activeIndices[i] < j) {
			m_calamus->swapElement(m_activeIndices[i], j);
			m_activeIndices[i] = j;
		}
	}

	m_calamus->setIndex(m_numFeather);
	computeFaceCalamusIndirection();
	discardActive();
	std::cout<<" reduce to "<<numFeathers();
}

void MlSkin::computeFaceCalamusIndirection()
{
	if(m_numFeather > 1)
		QuickSort::Sort(*m_calamus, 0, m_numFeather - 1);
		
	unsigned cur;
	unsigned pre = bodyMesh()->getNumFaces();
	for(unsigned i = 0; i < m_numFeather; i++) {
		cur = getCalamus(i)->faceIdx();
		if(cur != pre) {
			m_faceCalamusTable[cur].dartBegin = i;
			if(pre < bodyMesh()->getNumFaces())
				m_faceCalamusTable[pre].dartEnd = i;
			pre = cur;
		}
	}
	if(pre < bodyMesh()->getNumFaces())
		m_faceCalamusTable[pre].dartEnd = m_numFeather;
}

void MlSkin::resetFaceCalamusIndirection()
{
	for(unsigned i = 0; i < bodyMesh()->getNumFaces(); i++) m_faceCalamusTable[i].reset(i);
}

unsigned MlSkin::lastInactive(unsigned last) const
{	
	unsigned i = 0;	
	for(i = last; i > 0; i--) {
		if(!isActiveFeather(i)) return i;
	}
	return i;
}

bool MlSkin::isPointTooCloseToExisting(const Vector3F & pos, float minDistance)
{
	unsigned featherBegin, featherEnd;
	Vector3F d, p;
	for(unsigned i=0; i < numRegionElements(); i++) {
		featherBegin = m_faceCalamusTable[regionElementIndex(i)].dartBegin;
		featherEnd = m_faceCalamusTable[regionElementIndex(i)].dartEnd;
		for(unsigned j = featherBegin; j < featherEnd; j++) {
			MlCalamus *c = getCalamus(j);
			if(c->faceIdx() != regionElementIndex(i)) break;
			
			if(j < numFeathers()) {
				getPointOnBody(c, p);
			
				d = p - pos;
				if(d.length() < minDistance) return true;
			}
		}
	}

	return false;
}

bool MlSkin::isDartCloseToExisting(const Vector3F & pos, const std::vector<Vector3F> & existing, float minDistance) const
{
	unsigned dartBegin, dartEnd, i, j;
	for(i=0; i < numRegionElements(); i++) {
		if(!isFloodFace(regionElementIndex(i), dartBegin, dartEnd)) continue;
		for(j = dartBegin; j < dartEnd; j++) {
			if(Vector3F(pos, existing[j]).length() < minDistance) return true;
		}
	}
	return false;
}

bool MlSkin::isFloodFace(unsigned idx, unsigned & dartBegin, unsigned & dartEnd) const
{
	for(unsigned i = 0; i < m_floodFaces.size(); i++) {
		if(m_floodFaces[i].faceIdx == idx) {
			dartBegin = m_floodFaces[i].dartBegin;
			dartEnd = m_floodFaces[i].dartEnd;
			return true;
		}
	}
	return false;
}

bool MlSkin::isActiveFace(unsigned idx, std::vector<unsigned> & dartIndices) const
{
	unsigned dartBegin, dartEnd, i, j;
	for(i = 0; i < m_activeFaces.size(); i++) {
		if(m_activeFaces[i].faceIdx == idx) {
			dartBegin = m_activeFaces[i].dartBegin;
			dartEnd = m_activeFaces[i].dartEnd;
			for(j = dartBegin; j < dartEnd; j++)
				dartIndices.push_back(m_activeIndices[j]);
		}
	}
	return dartIndices.size() > 0;
}

bool MlSkin::isActiveFeather(unsigned idx) const
{
	MlCalamus *c = getCalamus(idx);
	unsigned i;
	std::vector<unsigned> activeInFace;
	if(!isActiveFace(c->faceIdx(), activeInFace)) return false;
	for(i = 0; i < activeInFace.size(); i++) {
		if(activeInFace[i] == idx) return true;
	}
	return false;
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
	bodyMesh()->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
}

void MlSkin::getNormalOnBody(MlCalamus * c, Vector3F &p) const
{
	bodyMesh()->normalOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
}

void MlSkin::tangentSpace(MlCalamus * c, Matrix33F & frm) const
{
	bodyMesh()->tangentFrame(c->faceIdx(), c->patchU(), c->patchV(), frm);
}

void MlSkin::rotationFrame(MlCalamus * c, const Matrix33F & tang, Matrix33F & frm) const
{
	frm.setIdentity();
	frm.rotateX(c->rotateX());
	frm.multiply(tang);
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

MlCalamusArray * MlSkin::getCalamusArray() const
{
	return m_calamus;
}

void MlSkin::resetFloodFaces()
{
	m_floodFaces.clear();
	for(unsigned i = 0; i < numRegionElements(); i++)
		m_floodFaces.push_back(FloodTable(regionElementIndex(i)));
}

void MlSkin::restFloodFacesAsActive()
{
	m_floodFaces.clear();
	for(unsigned i = 0; i < numActiveRegionFaces(); i++)
		m_floodFaces.push_back(FloodTable(activeRegionFace(i)));
}

void MlSkin::verbose() const
{
	std::cout<<"face id\n";
	for(unsigned i = 0; i < m_numFeather; i++) {
		std::cout<<" "<<getCalamus(i)->faceIdx();
	}
	std::cout<<"\n";
	
	std::cout<<"face start\n";
	for(unsigned i = 0; i < bodyMesh()->getNumFaces(); i++) {
		if(m_faceCalamusTable[i].dartEnd > 0) std::cout<<" "<<i<<":"<<m_faceCalamusTable[i].dartBegin;
	}
	std::cout<<"\n";
}
//~:
