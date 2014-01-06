/*
 *  DeformableFeather.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/6/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "DeformableFeather.h"
#include "BaseVane.h"
DeformableFeather::DeformableFeather() 
{
	m_group = 0;
}

DeformableFeather::~DeformableFeather() 
{
	if(m_group) delete[] m_group;
}

void DeformableFeather::computeTexcoord()
{
	TexturedFeather::computeTexcoord();
	computeBinding();
}

void DeformableFeather::computeBinding()
{
	if(m_group) delete[] m_group;
	
	const short ns = numSegment();
	
	m_group = new BindGroup[ns];
	
	for(short i=0; i<ns; i++) {
		m_group[i]._bind = new BindCoord[(uvVane(0)->gridU()+1) * (uvVane(0)->gridV()+1)];
		m_group[i]._numBind = 0;
	}
	bindVane(uvVane(0), 0);
	bindVane(uvVane(1), 1);
}

void DeformableFeather::bindVane(BaseVane * vane, short rgt)
{
	float segX, minY, maxY;
	const short ns = numSegment();
	for(unsigned j = 0; j <= vane->gridU(); j++) {
		for(unsigned k = 0; k <= vane->gridV(); k++) {
			Vector3F * cv = vane->railCV(j, k);

			for(short i=0; i < ns; i++) {
				segX = (*segmentQuillTexcoord(i)).x;
				minY = (*segmentQuillTexcoord(i)).y;
				maxY = (*segmentQuillTexcoord(i+1)).y;
				if(i == ns - 1) maxY += 10e8;
				if(cv->y < maxY) {
					BindCoord & coord = m_group[i]._bind[m_group[i]._numBind];
					coord._rgt = rgt;
					coord._u = j;
					coord._v = k;
					coord._taper = (cv->x - segX) / width();
					if(coord._taper < 0.f) coord._taper = -coord._taper;
					coord._objP = Vector3F(cv->x - segX, cv->y - minY, 0.f);
					m_group[i]._numBind++;
					break;
				}
			}
		}
	}
}

short DeformableFeather::numBind(short seg) const
{
	return m_group[seg]._numBind;
}

Vector3F DeformableFeather::getBind(short seg, short idx, short & u, short & v, short & side, float & taper) const
{
	BindCoord & coord = m_group[seg]._bind[idx];
	u = coord._u;
	v = coord._v;
	taper = coord._taper;
	side = coord._rgt;
	return coord._objP;
}
