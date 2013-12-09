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
#include <BaseImage.h>
#include <SelectCondition.h>
#include <FloodCondition.h>
#include "MlCalamusArray.h"
#include "MlCluster.h"

MlSkin::MlSkin() : m_numCreatedFeather(0)
{
    m_activeIndices.clear();
}

MlSkin::~MlSkin()
{
}

void MlSkin::setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo)
{
	CollisionRegion::setBodyMesh(mesh, topo);
	createFaceCalamusIndirection();
	resetFaceCalamusIndirection();
	createFaceVicinity();
	createFaceCluster();
	if(!bodyMesh()->hasVertexData("weishell")) {
		float * disw = bodyMesh()->perVertexFloat("weishell");
		const unsigned nv = bodyMesh()->getNumVertices();
		for(unsigned i = 0; i < nv; i++) disw[i] = .5f;
	}
	CollisionRegion::useRegionElementVertexVector("aftshell");
}

void MlSkin::floodAround(MlCalamus floodC, FloodCondition * condition)
{	
	unsigned i, j, faceInd;
	
	float u, v;
	Vector3F adart, facing;
	std::vector<Vector3F> darts;
	for(i = 0; i < m_floodFaces.size(); i++) {
		faceInd = m_floodFaces[i].faceIdx;
		m_floodFaces[i].dartBegin = m_floodFaces[i].dartEnd = darts.size();
		const unsigned ndart = 4 + bodyMesh()->calculateBBox(faceInd).area() / condition->minDistance() / condition->minDistance();
		for(j = 0; j < ndart; j++) {
			if(condition->filteredByProbability()) continue;
		
			u = ((float)(rand()%591))/591.f;
			v = ((float)(rand()%593))/593.f;
			bodyMesh()->pointOnPatch(faceInd, u, v, adart);
			
			if(condition->byDistance()) {
				if(condition->filteredByDistance(adart)) continue;
			}
			
			if(hasActiveRegion() && condition->byRegion()) {
				if(!sampleColorMatches(faceInd, u, v)) continue;
			}
			
			//bodyMesh()->normalOnPatch(faceInd, u, v, facing);
			//if(facing.dot(floodNor) < .23f) continue;
			resetCollisionRegion(faceInd);
	
			if(isPointTooCloseToExisting(adart, condition->minDistance())) continue;
			if(isDartCloseToExisting(adart, darts, condition->minDistance())) continue;
				
			darts.push_back(adart);
			floodC.bindToFace(faceInd, u, v);
			createFeather(floodC);
			m_floodFaces[i].dartEnd++;
		}
	}
	darts.clear();
}

void MlSkin::selectAround(unsigned idx, SelectCondition * selcon)
{	
	resetCollisionRegionByDistance(idx, selcon->center(), selcon->maxDistance());
	
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
	unsigned featherBegin, featherEnd;
	faceCalamusBeginEnd(faceIdx, featherBegin, featherEnd);
	for(unsigned j = featherBegin; j < featherEnd; j++) {
		if(j >= numFeathers()) break;
	    
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
	addFeather(ori);
	
	m_activeIndices.push_back(numFeathers() - 1);
	m_numCreatedFeather++;
	
	return true;
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

	if(direction.dot(activeMeanDir) < 0.f) activeMeanPitch -= .1f;
	else activeMeanPitch += .1f;
	
	for(i =0; i < num; i++) {
		MlCalamus * c = getActive(i);
		
		getPointOnBody(c, p);
		drop = Vector3F(p, center).length() / radius;
		drop = 1.f - drop * drop;

		c->setRotateY(activeMeanPitch * drop + c->rotateY() * (1.f - drop));
    }
}

void MlSkin::smoothShell(const Vector3F & center, const float & radius, const float & weight)
{
	const float increase = (weight - 1.f) * 0.3f;
	float * disw = bodyMesh()->perVertexFloat("weishell");
	Vector3F d;
	float l;
	std::vector<unsigned> vertices;
	regionElementVertices(vertices);
	std::vector<unsigned>::const_iterator it = vertices.begin();
	for(; it != vertices.end(); ++it) {
		Vector3F d = bodyMesh()->getVertices()[*it] - center;
		l = d.length();
		if(l < radius) {
			disw[*it] += increase;
			if(disw[*it] < 0.f) disw[*it] = 0.f;
			else if(disw[*it] > 1.f) disw[*it] = 1.f;
		}
	}

	vertices.clear();
	computeVertexDisplacement();
}

void MlSkin::finishCreateFeather()
{
	if(!hasFeatherCreated()) return;
	if(!bodyMesh()) return;
    computeFaceCalamusIndirection();
	conputeFaceClustering();
	m_numCreatedFeather = 0;
	std::cout<<" increase to "<<numFeathers();
}

void MlSkin::finishEraseFeather()
{
	if(numActive() < 1) return;
	resetFaceCalamusIndirection();
		
	if(numActive() == numFeathers()) {
		zeroFeather();	
		std::cout<<" reduce to 0";
		return;
	}
	
	unsigned i, j = numFeathers() - 1;
	reduceFeather(numActive());
	
	MlCalamusArray * ca = getCalamusArray();
	const unsigned num = numActive();
	for(i = 0; i < num; i++) {
		if(m_activeIndices[i] >= numFeathers()) continue;
		j = lastInactive(j);
		if(m_activeIndices[i] < j) {
			ca->swapElement(m_activeIndices[i], j);
			m_activeIndices[i] = j;
		}
	}

	computeFaceCalamusIndirection();
	conputeFaceClustering();
	discardActive();
	std::cout<<" reduce to "<<numFeathers();
}

unsigned MlSkin::lastInactive(unsigned last) const
{	
	unsigned i = 0;	
	for(i = last; i > 0; i--) {
		if(!isActiveFeather(i)) return i;
	}
	return i;
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

unsigned MlSkin::numActive() const
{
	return m_activeIndices.size();
}

MlCalamus * MlSkin::getActive(unsigned idx) const
{
	return getCalamus(m_activeIndices[idx]);
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
	return getCalamus(numFeathers() - m_numCreatedFeather + idx);
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

void MlSkin::computeVertexDisplacement()
{
	const unsigned nv = bodyMesh()->getNumVertices();
	Vector3F * dis1 = bodyMesh()->perVertexVector("aftshell");
	float * disw = bodyMesh()->perVertexFloat("weishell");
	
	Vector3F * nor = bodyMesh()->getNormals();
	
	topology()->calculateSmoothedNormal(dis1);
	
	for(unsigned i = 0; i < nv; i++) {
		dis1[i] = dis1[i] * (1.f - disw[i]) + nor[i] * disw[i];
		dis1[i].normalize();
	}
}

void MlSkin::shellUp(std::vector<Vector3F> & dst)
{
	const unsigned n = numActive();
	Vector3F p, u, d;
	for(unsigned i=0; i < n; i++) {
		MlCalamus * c = getActive(i);
		getPointOnBody(c, p);
		dst.push_back(p);
		interpolateVertexVector(c->faceIdx(), c->patchU(), c->patchV(), &d);
		//u += d;
		//
		Matrix33F space, tang;
		tangentSpace(c, tang);
		rotationFrame(c, tang, space);
		Vector3F f = space.transform(Vector3F::ZAxis);
		u = p + f * c->realScale();
		dst.push_back(u);
		
		dst.push_back(u);
		u = getClosestNormal(u, 1000.f, p);
		dst.push_back(p);
		dst.push_back(p);
		dst.push_back(p + u);
	}
}

void MlSkin::verbose() const
{
	std::cout<<"face id\n";
	for(unsigned i = 0; i < numFeathers(); i++) {
		std::cout<<" "<<getCalamus(i)->faceIdx();
	}
	std::cout<<"\n";
	
	std::cout<<"face start\n";
	for(unsigned i = 0; i < bodyMesh()->getNumFaces(); i++) {
		//if(m_faceCalamusTable[i].dartEnd > 0) std::cout<<" "<<i<<":"<<m_faceCalamusTable[i].dartBegin;
	}
	std::cout<<"\n";
}
//~:
