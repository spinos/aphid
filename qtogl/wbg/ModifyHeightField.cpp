/*
 *  ModifyHeightField.cpp
 *  
 *
 *  Created by jian zhang on 3/25/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ModifyHeightField.h"
#include "DrawHeightField.h"
#include <img/HeightField.h>
#include <ttg/GlobalElevation.h>
#include <math/Plane.h>
#include <cmath>
#include "wbg_common.h"

using namespace aphid;

ModifyHeightField::ModifyHeightField()
{
	m_heightDrawer = new DrawHeightField;
	m_heightFieldToolFlag = 0;
	m_plane = new Plane(0.f, 1.f, 0.f, -m_heightDrawer->planeHeight() );
}

ModifyHeightField::~ModifyHeightField()
{}

void ModifyHeightField::setHeightFieldToolFlag(int x)
{ m_heightFieldToolFlag = x; }

const int & ModifyHeightField::heightFieldToolFlag() const
{ return m_heightFieldToolFlag; }

void ModifyHeightField::drawHeightField()
{
	const int n = ttg::GlobalElevation::NumHeightFields();
	for(int i=0;i<n;++i) {
		 const img::HeightField & fld = ttg::GlobalElevation::GetHeightField(i);
		 m_heightDrawer->drawBound(fld);
		 if(m_heightDrawer->curFieldInd() == i) {
			m_heightDrawer->drawValue(fld);
		 }
	}
}

void ModifyHeightField::selectHeightField(int x)
{
	const img::HeightField & fld = ttg::GlobalElevation::GetHeightField(x);
	if(m_heightDrawer->curFieldInd() != x) {
		m_heightDrawer->setCurFieldInd(x);
		m_heightDrawer->bufferValue(fld);
	}
}

bool ModifyHeightField::updateActiveState(Vector2F & wldv, Vector2F & objv,
						const Ray * incident)
{
	if(m_heightDrawer->curFieldInd() < 0) {
		return false;
	}
	
	Vector3F p3;
	float t;
	m_plane->rayIntersect(*incident, p3, t);
	
	Vector2F v2(p3.x, p3.z);
	const img::HeightField & fld = ttg::GlobalElevation::GetHeightField(m_heightDrawer->curFieldInd() );
	m_isActive = fld.getLocalPnt(v2);
	if(!m_isActive) {
		return false;
	}
	
	wldv.set(p3.x, p3.z);
	objv = v2;
	
	m_worldCenterP = fld.worldCenterPnt();
	return true;
}

void ModifyHeightField::beginModifyHeightField(const Ray * incident)
{
	Vector2F wp, op;
	if(!updateActiveState(wp, op, incident) ) {
		return;
	}
	
	m_lastWorldP = wp;
	m_lastLocalP = op;
}
	
void ModifyHeightField::endModifyeHeightField()
{
	m_isActive = false;
}

void ModifyHeightField::doMoveHeightField(const Ray * incident)
{
	Vector2F wp, op;
	if(!updateActiveState(wp, op, incident) ) {
		return;
	}
	
	const Vector2F dwp = wp - m_lastWorldP;
	img::HeightField * fld = ttg::GlobalElevation::HeightFieldR(m_heightDrawer->curFieldInd() );
	Matrix44F space = fld->transformMatrix();
	space.translate(dwp.x, 0.f, dwp.y);
	fld->setTransformMatrix(space);
	
	m_lastWorldP = wp;
	m_lastLocalP = op;
}

void ModifyHeightField::doRotateHeightField(const aphid::Ray * incident)
{
	Vector2F wp, op;
	if(!updateActiveState(wp, op, incident) ) {
		return;
	}
				
	Vector3F va(m_lastWorldP.x - m_worldCenterP.x,
				0.f,
				m_lastWorldP.y - m_worldCenterP.y); 
	va.normalize();
	Vector3F vb(wp.x - m_worldCenterP.x,
				0.f,
				wp.y - m_worldCenterP.y);
	vb.normalize();
	
	float ang = acos(va.dot(vb));
	if(va.cross(vb).y < 0.f) {
		ang = -ang;
	}
	
	img::HeightField * fld = ttg::GlobalElevation::HeightFieldR(m_heightDrawer->curFieldInd() );
	Matrix44F space = fld->transformMatrix();
	space.rotateY(ang);
	
	const Vector2F lc = fld->localCenterPnt();
	Vector3F dc(lc.x, 0.f, lc.y);
	dc = space.transform(dc);
	dc.x = m_worldCenterP.x - dc.x;
	dc.z = m_worldCenterP.y - dc.z;
	space.translate(dc);
	
	fld->setTransformMatrix(space);
	
	m_lastWorldP = wp;
	m_lastLocalP = op;
}

void ModifyHeightField::doResizeHeightField(const aphid::Ray * incident)
{
	Vector2F wp, op;
	if(!updateActiveState(wp, op, incident) ) {
		return;
	}
	
	Vector3F va(m_lastWorldP.x - m_worldCenterP.x,
				0.f,
				m_lastWorldP.y - m_worldCenterP.y); 
	
	Vector3F vb(wp.x - m_worldCenterP.x,
				0.f,
				wp.y - m_worldCenterP.y);
	
	float sfac = vb.length() / va.length();
	
	img::HeightField * fld = ttg::GlobalElevation::HeightFieldR(m_heightDrawer->curFieldInd() );
	Matrix44F space = fld->transformMatrix();
	
	Vector3F scl = space.scale();
	if(scl.x < 1.f) {
		scl.x = 1.f/scl.x;
		scl.y = 1.f/scl.y;
		scl.z = 1.f/scl.z;
	} else {
		scl.x = sfac;
		scl.y = 1.f;
		scl.z = sfac;
	}
	space.scaleBy(scl);
	
	const Vector2F lc = fld->localCenterPnt();
	Vector3F dc(lc.x, 0.f, lc.y);
	dc = space.transform(dc);
	dc.x = m_worldCenterP.x - dc.x;
	dc.z = m_worldCenterP.y - dc.z;
	space.translate(dc);
	
	fld->setTransformMatrix(space);
	
	m_lastWorldP = wp;
	m_lastLocalP = op;
}
	