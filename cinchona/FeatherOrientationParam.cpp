/*
 *  FeatherOrientationParam.cpp
 *  cinchona
 *
 *  Created by jian zhang on 1/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "FeatherOrientationParam.h"

#include <math/Matrix33F.h>
#include <gpr/GPInterpolate.h>

using namespace aphid;

FeatherOrientationParam::FeatherOrientationParam()
{
	m_rots = new Matrix33F[4];
	std::memset(m_rots, 0, 4 * 36);
	m_vecs = new Vector3F[8];
	std::memset(m_vecs, 0, 8 * 12);
	m_sideInterp = new gpr::GPInterpolate<float>();
	m_sideInterp->create(4,1,3);
	m_sideInterp->setFilterLength(.5f);
	m_upInterp = new gpr::GPInterpolate<float>();
	m_upInterp->create(4,1,3);
	m_upInterp->setFilterLength(.5f);
}

FeatherOrientationParam::~FeatherOrientationParam()
{
	delete[] m_vecs;
	delete[] m_rots;
	delete m_sideInterp;
	delete m_upInterp;
}

void FeatherOrientationParam::set(const Matrix33F * mats)
{
	m_changed = false;
	Vector3F side, up, front;
	for(int i=0;i<4;++i) {
		mats[i].getSide(side);
		mats[i].getUp(up);
		mats[i].getFront(front);
	
		if(*rotationSideR(i) != side) {
			m_changed = true;
			*rotationSideR(i) = side;
		}
		if(*rotationUpR(i) != up) {
			m_changed = true;
			*rotationUpR(i) = up;
		}
		
		side.normalize();
		up.normalize();
		front.normalize();
		m_rots[i].fill(side, up, front);
		
	}
	
	if(m_changed) {
		learnOrientation();
	}
}

bool FeatherOrientationParam::isChanged() const
{ return m_changed; }

Vector3F * FeatherOrientationParam::rotationSideR(int i)
{ return &m_vecs[i<<1]; }

Vector3F * FeatherOrientationParam::rotationUpR(int i)
{ return &m_vecs[(i<<1)+1]; }

const Matrix33F & FeatherOrientationParam::rotation(int i) const
{ return m_rots[i]; }

void FeatherOrientationParam::learnOrientation()
{
	float vx[4] = {0.99f, .67f, .33f, .01f};
	for(int i=0;i<4;++i) {
		m_sideInterp->setObservationi(i, &vx[i], (const float *)rotationSideR(i));
		m_upInterp->setObservationi(i, &vx[i], (const float *)rotationUpR(i));
	}
	
	if(!m_sideInterp->learn() ) {
		std::cout<<"FeatherOrientationParam learnOrientation side interpolate failed to learn";
	}
	
	if(!m_upInterp->learn() ) {
		std::cout<<"FeatherOrientationParam learnOrientation up interpolate failed to learn";
	}
}

void FeatherOrientationParam::predictRotation(aphid::Matrix33F & dst,
						const float * x)
{
	m_sideInterp->predict(x);
	m_upInterp->predict(x);
	
	const float * sideY = m_sideInterp->predictedY().column(0);
	const float * upY = m_upInterp->predictedY().column(0);
	
	Vector3F vside(sideY[0], sideY[1], sideY[2]);
	vside.normalize();
	Vector3F vup(upY[0], upY[1], upY[2]);
	vup.normalize();
	
	Vector3F vfront = vside.cross(vup);
	
	vside = vup.cross(vfront);
	vside.normalize();
	
	dst.fill(vside, vup, vfront);
}

gpr::GPInterpolate<float > * FeatherOrientationParam::sideInterp()
{ return m_sideInterp; }
	
gpr::GPInterpolate<float > * FeatherOrientationParam::upInterp()
{ return m_upInterp; }
