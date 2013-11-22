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
#include "MlCalamusArray.h"

MlSkin::MlSkin() : m_numFeather(0), m_faceCalamusStart(0), m_numCreatedFeather(0)
{
    m_activeIndices.clear();
	m_calamus = new MlCalamusArray; 
	m_floodCondition = ByDistance;
	m_floodRegion = m_eraseRegion = 0;
}

MlSkin::~MlSkin()
{
	cleanup();
}

void MlSkin::cleanup()
{
	clearFeather();
	if(m_faceCalamusStart) {
		delete[] m_faceCalamusStart;
		m_faceCalamusStart = 0;
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
	m_faceCalamusStart = new unsigned[mesh->getNumFaces()];
	resetFaceCalamusIndirection();
}

void MlSkin::floodAround(MlCalamus floodC, const Vector3F & floodPos, const Vector3F & floodNor, const float & floodMaxD, const float & floodMinD)
{	
	unsigned i, j, iface;
	
	char outsideMaxD;
	float u, v;
	Vector3F adart, facing;
	std::vector<Vector3F> darts;
	for(i = 0; i < m_floodFaces.size(); i++) {
		iface = m_floodFaces[i].faceIdx;
		m_floodFaces[i].dartBegin = m_floodFaces[i].dartEnd = darts.size();
		const unsigned ndart = 4 + bodyMesh()->calculateBBox(iface).area() / floodMinD / floodMinD;
		for(j = 0; j < ndart; j++) {
		
			u = ((float)(rand()%591))/591.f;
			v = ((float)(rand()%593))/593.f;
			bodyMesh()->pointOnPatch(iface, u, v, adart);
			
			outsideMaxD = Vector3F(floodPos, adart).length() > floodMaxD;
			if(m_floodCondition == ByDistance) {
				if(outsideMaxD) continue;
			}
			else {
				if(!m_floodRegion) {
					if(outsideMaxD) continue;
				}
				if(!sampleColorMatches(iface, u, v)) continue;
			}
			
			//bodyMesh()->normalOnPatch(iface, u, v, facing);
			//if(facing.dot(floodNor) < .23f) continue;
			resetCollisionRegion(iface);
	
			if(isPointTooCloseToExisting(adart, floodMinD)) continue;
			
			if(isDartCloseToExisting(adart, darts, floodMinD)) continue;
				
			darts.push_back(adart);
			floodC.bindToFace(iface, u, v);
			createFeather(floodC);
			m_floodFaces[i].dartEnd++;
		}
	}
	darts.clear();
}

void MlSkin::selectAround(unsigned idx, const Vector3F & pos, const Vector3F & nor, const float & maxD, char byRegion)
{	
	m_selectCondition.center = pos;
	m_selectCondition.normal = nor;
	m_selectCondition.maxDistance = maxD;
	m_selectCondition.byRegion = byRegion;
	
	resetCollisionRegionAround(idx, pos, maxD);
	
	for(unsigned i=0; i < numRegionElements(); i++)
		selectFeatherByFace(regionElementIndex(i));
}

void MlSkin::selectFeatherByFace(unsigned faceIdx)
{
	Vector3F d, p, n;
	
	const unsigned maxCountPerFace = numFeathers();
	unsigned ifeather = m_faceCalamusStart[faceIdx];
	for(unsigned j = 0; j < maxCountPerFace; j++) {
		if(ifeather >= m_numFeather) return;
				
		MlCalamus *c = getCalamus(ifeather);
		if(c->faceIdx() != faceIdx) return;
		
		getNormalOnBody(c, n);
		if(n.dot(m_selectCondition.normal) < 0.f) {
			ifeather++;
			continue;
		}
		
		getPointOnBody(c, p);
		
		d = p - m_selectCondition.center;
		if(d.length() > m_selectCondition.maxDistance) {
			ifeather++;
			continue;
		}
			
		if(hasActiveFaces() && m_selectCondition.byRegion) {
			if(!sampleColorMatches(c->faceIdx(), c->patchU(), c->patchV())) {
				ifeather++;
				continue;
			}
		}
		
		if(!IsElementIn(ifeather, m_activeIndices))
			m_activeIndices.push_back(ifeather);
		
		ifeather++;
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

	if(direction.dot(activeMeanDir) < 0.f) activeMeanPitch -= .07f;
	else activeMeanPitch += .07f;
	
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
	std::cout<<" add to "<<numFeathers();
}

void MlSkin::finishEraseFeather()
{
	if(numActive() < 1) return;
	if(numActive() == numFeathers()) {
		m_numFeather = 0;
		m_calamus->setIndex(0);
		resetFaceCalamusIndirection();
		std::cout<<" reduce all 0";
		return;
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
			m_faceCalamusStart[cur] = i;
			pre = cur;
		}
	}
}

void MlSkin::resetFaceCalamusIndirection()
{
	for(unsigned i = 0; i < bodyMesh()->getNumFaces(); i++) m_faceCalamusStart[i] = 0;
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

bool MlSkin::isPointTooCloseToExisting(const Vector3F & pos, float minDistance)
{
	const unsigned maxCountPerFace = m_numFeather / 2;
	
	Vector3F d, p;
	for(unsigned i=0; i < numRegionElements(); i++) {
		
		unsigned ifeather = m_faceCalamusStart[regionElementIndex(i)];
		for(unsigned j = 0; j < maxCountPerFace; j++) {
			MlCalamus *c = getCalamus(ifeather);
			if(c->faceIdx() != regionElementIndex(i)) break;
			
			if(ifeather < numFeathers()) {
				getPointOnBody(c, p);
			
				d = p - pos;
				if(d.length() < minDistance) return true;
			}
			
			ifeather++;
		}
	}

	return false;
}

bool MlSkin::isDartCloseToExisting(const Vector3F & pos, const std::vector<Vector3F> & existing, float minDistance) const
{
	unsigned dartBegin, dartEnd, i, j;
	for(i=0; i < numRegionElements(); i++) {
		if(!isFloodFace(regionElementIndex(i), dartBegin, dartEnd)) continue;
		if(dartEnd == dartBegin) continue;
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

void MlSkin::setFloodCondition(FloodCondition fc)
{
	m_floodCondition = fc;
}

MlSkin::FloodCondition MlSkin::floodCondition() const
{
	return m_floodCondition;
}

char MlSkin::hasActiveFaces() const
{
    return m_activeFaces.size() > 0;
}

void MlSkin::clearActiveFaces()
{
    m_activeFaces.clear();
}

void MlSkin::resetActiveFaces()
{
	m_activeFaces.clear();
	for(unsigned i = 0; i < numRegionElements(); i++)
		m_activeFaces.push_back(regionElementIndex(i));
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
	for(unsigned i = 0; i < m_activeFaces.size(); i++)
		m_floodFaces.push_back(FloodTable(m_activeFaces[i]));
}

void MlSkin::setFloodRegion(char on)
{
	m_floodRegion = on;
}

char MlSkin::floodRegion() const
{
	return m_floodRegion;
}

void MlSkin::setEraseRegion(char on)
{
    m_eraseRegion = on;
}

char MlSkin::eraseRegion() const
{
    return m_eraseRegion;
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
		if(m_faceCalamusStart[i] > 0) std::cout<<" "<<i<<":"<<m_faceCalamusStart[i];
	}
	std::cout<<"\n";
}
//~:
