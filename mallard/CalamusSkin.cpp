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
CalamusSkin::CalamusSkin() : m_numFeather(0), m_perFaceVicinity(0)
{
	m_calamus = new MlCalamusArray;
}

CalamusSkin::~CalamusSkin() {}

void CalamusSkin::clearFaceVicinity()
{
	if(m_perFaceVicinity) {
		delete[] m_perFaceVicinity;
		m_perFaceVicinity = 0;
	}
}

void CalamusSkin::createFaceVicinity()
{
	clearFaceVicinity();
	m_perFaceVicinity = new float[bodyMesh()->getNumFaces()];
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

void CalamusSkin::resetFaceVicinity()
{
    const unsigned nf = bodyMesh()->getNumFaces();
	for(unsigned i= 0; i < nf; i++) m_perFaceVicinity[i] = 0.f;
}

void CalamusSkin::setFaceVicinity(unsigned idx, float val)
{
    m_perFaceVicinity[idx] = val;
}

float CalamusSkin::faceVicinity(unsigned idx) const
{
    return m_perFaceVicinity[idx];
}