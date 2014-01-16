/*
 *  CalamusSkin.cpp
 *  mallard
 *
 *  Created by jian zhang on 12/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "CalamusSkin.h"
#include <AccPatchMesh.h>
#include <MlCalamus.h>
#include <MlCalamusArray.h>
#include "MlCluster.h"
#include <QuickSort.h>
#include <BaseSphere.h>
CalamusSkin::CalamusSkin() : m_numFeather(0), m_faceCalamusTable(0)
{
	m_calamus = new MlCalamusArray;
}

CalamusSkin::~CalamusSkin() { cleanup(); }

void CalamusSkin::cleanup()
{
	clearFeather();
	if(m_faceCalamusTable) delete[] m_faceCalamusTable;
	m_faceCalamusTable = 0;
	m_perFaceCluster.reset();
}

void CalamusSkin::setBodyMesh(AccPatchMesh * mesh)
{
	CollisionRegion::setBodyMesh(mesh);
	createFaceCalamusIndirection();
	resetFaceCalamusIndirection();
	createFaceCluster();
}

void CalamusSkin::getPointOnBody(MlCalamus * c, Vector3F &p) const
{
	bodyMesh()->pointOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
}

void CalamusSkin::getNormalOnBody(MlCalamus * c, Vector3F &p) const
{
	bodyMesh()->normalOnPatch(c->faceIdx(), c->patchU(), c->patchV(), p);
}

void CalamusSkin::tangentSpace(MlCalamus * c, Matrix33F & frm) const
{
	bodyMesh()->tangentFrame(c->faceIdx(), c->patchU(), c->patchV(), frm);
}

void CalamusSkin::rotationFrame(MlCalamus * c, const Matrix33F & tang, Matrix33F & frm) const
{
	frm.setIdentity();
	frm.rotateX(c->rotateX());
	frm.multiply(tang);
}

void CalamusSkin::calamusSpace(MlCalamus * c, Matrix33F & frm) const
{
	Matrix33F tang;
	tangentSpace(c, tang);
	rotationFrame(c, tang, frm);
}

MlCalamusArray * CalamusSkin::getCalamusArray() const
{
	return m_calamus;
}

MlCalamus * CalamusSkin::getCalamus(unsigned idx) const
{
	return m_calamus->asCalamus(idx);
}

void CalamusSkin::clearFeather()
{
	m_calamus->clear();
	m_calamus->initialize();
	m_numFeather = 0;
}

void CalamusSkin::setNumFeathers(unsigned num)
{
	m_calamus->expandBy(num);
	m_numFeather = num;
}

unsigned CalamusSkin::numFeathers() const
{
	return m_numFeather;
}

void CalamusSkin::addFeather(MlCalamus & ori)
{
	m_calamus->expandBy(1);
	MlCalamus * c = m_calamus->asCalamus();
	*c = ori;
	m_calamus->next();

	m_numFeather++;
}

void CalamusSkin::zeroFeather()
{
	m_calamus->setIndex(0);
	m_numFeather = 0;
}

void CalamusSkin::reduceFeather(unsigned num)
{
	m_numFeather -= num;
	m_calamus->setIndex(m_numFeather);
}

void CalamusSkin::touchBy(MlCalamus * c, const Vector3F & pos, const Matrix33F & frm)
{
	const unsigned fi = c->faceIdx();
	const float fv = c->realScale() * 0.5f;
	Vector3F d = frm.transform(Vector3F::ZAxis);
	d = pos + d * fv;
	BaseSphere sph;
	sph.setCenter(d);
	sph.setRadius(fv);
	resetCollisionRegionAround(fi, sph);
}

void CalamusSkin::createFaceCluster()
{
	m_perFaceCluster.reset(new MlCluster[bodyMesh()->getNumFaces()]);
}

void CalamusSkin::computeFaceClustering()
{
	const unsigned nf = bodyMesh()->getNumFaces();
	for(unsigned i = 0; i < nf; i++)
		m_perFaceCluster[i].compute(m_calamus, bodyMesh(), m_faceCalamusTable[i].dartBegin, m_faceCalamusTable[i].dartEnd);
}

void CalamusSkin::computeClusterSamples()
{
	unsigned nk;
	Vector3F p;
	Matrix33F space;
	const unsigned nf = bodyMesh()->getNumFaces();
	for(unsigned i = 0; i < nf; i++) { 
		MlCluster & cluster = m_perFaceCluster[i];
		if(!cluster.isValid()) continue;
		nk = cluster.K(); 
		for(unsigned j  =0; j < nk; j++) {
			MlCalamus * c = getCalamus(cluster.sampleIdx(j));
			getPointOnBody(c, p);
			calamusSpace(c, space);
			touchBy(c, p, space);
			c->bendFeather(p, space);
			cluster.recordAngles(c, j);
		}
		//cluster.computeSampleDirs(m_calamus, bodyMesh());
	}
}

char CalamusSkin::useClusterSamples(unsigned faceIdx, unsigned perFaceIdx, MlCalamus * c, unsigned ci)
{
	MlCluster & cluster = m_perFaceCluster[faceIdx];
	const unsigned grp = cluster.group(perFaceIdx);
	if(cluster.sampleIdx(grp) == ci) {
		cluster.reuseAngles(c, grp);
		return 1;
	}
	
	if(cluster.sampleBend(grp) > -0.01f) return 0;
	
	const short nseg = cluster.sampleNSeg(grp);
	if(c->featherNumSegment() != nseg) return 0;
		
	cluster.reuseAngles(c, grp);
	return 1;
}

unsigned CalamusSkin::clusterK(unsigned faceIdx) const
{
	return m_perFaceCluster[faceIdx].K();
}

void CalamusSkin::getClustering(unsigned idx, std::vector<Vector3F> & dst)
{
	Vector3F p;
	for(unsigned i = m_faceCalamusTable[idx].dartBegin; i < m_faceCalamusTable[idx].dartEnd; i++) {
		MlCalamus * c = getCalamus(i);
		getPointOnBody(c, p);
		dst.push_back(p);
		
		p = m_perFaceCluster[idx].groupCenter(i - m_faceCalamusTable[idx].dartBegin);
		dst.push_back(p);
	}
}

void CalamusSkin::createFaceCalamusIndirection()
{ 
	m_faceCalamusTable = new FloodTable[bodyMesh()->getNumFaces()];
}

void CalamusSkin::resetFaceCalamusIndirection()
{
	for(unsigned i = 0; i < bodyMesh()->getNumFaces(); i++) m_faceCalamusTable[i].reset(i);
}

void CalamusSkin::computeFaceCalamusIndirection()
{
	const unsigned numFeather = numFeathers();
	if(numFeather > 1)
		QuickSort::Sort(*getCalamusArray(), 0, numFeather - 1);
		
	unsigned cur;
	unsigned pre = bodyMesh()->getNumFaces();
	for(unsigned i = 0; i < numFeather; i++) {
		cur = getCalamus(i)->faceIdx();
		if(cur != pre) {
			m_faceCalamusTable[cur].dartBegin = i;
			if(pre < bodyMesh()->getNumFaces())
				m_faceCalamusTable[pre].dartEnd = i;
			pre = cur;
		}
	}
	if(pre < bodyMesh()->getNumFaces())
		m_faceCalamusTable[pre].dartEnd = numFeather;
}

void CalamusSkin::faceCalamusBeginEnd(unsigned faceIdx, unsigned & begin, unsigned & end) const
{
	begin = m_faceCalamusTable[faceIdx].dartBegin;
	end = m_faceCalamusTable[faceIdx].dartEnd;
}

bool CalamusSkin::isPointTooCloseToExisting(const Vector3F & pos, float minDistance)
{
	unsigned featherBegin, featherEnd;
	Vector3F d, p;	
	for(unsigned i=0; i < numRegionElements(); i++) {
		faceCalamusBeginEnd(regionElementIndex(i), featherBegin, featherEnd);
		for(unsigned j = featherBegin; j < featherEnd; j++) {
			MlCalamus *c = getCalamus(j);
			if(j < numFeathers()) {
			    if(c->faceIdx() != regionElementIndex(i)) break;
				getPointOnBody(c, p);
			
				d = p - pos;
				if(d.length() < minDistance) return true;
			}
		}
	}

	return false;
}