/*
 *  DrawPoint.cpp
 *  
 *
 *  Created by jian zhang on 1/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawPoint.h"
#include <math/Vector3F.h>
#include <gl_heads.h>

namespace aphid {

DrawPoint::DrawPoint() :
m_pntBufLength(0)
{}
	
DrawPoint::~DrawPoint()
{}

const int & DrawPoint::pntBufLength() const
{ return m_pntBufLength; }

Vector3F * DrawPoint::pntNormalR()
{ return m_pntNormalBuf.get(); }
	
Vector3F * DrawPoint::pntPositionR()
{ return m_pntPositionBuf.get(); }

const float * DrawPoint::pntNormalBuf() const
{ return (const float *)m_pntNormalBuf.get(); }

const float * DrawPoint::pntPositionBuf() const
{ return (const float *)m_pntPositionBuf.get(); }

void DrawPoint::setPointDrawBufLen(const int & x)
{ 
	m_pntBufLength = x;
	if(x<1) {
		m_pntNormalBuf.reset();
		m_pntPositionBuf.reset();
	} else {
		m_pntNormalBuf.reset(new Vector3F[x]);
		m_pntPositionBuf.reset(new Vector3F[x]);
	}
}

void DrawPoint::buildPointDrawBuf(const int & nv,
				const float * vertP, 
				const float * vertN,
				int stride)
{
	setPointDrawBufLen(nv);
	if(nv < 1) {
		return;
	}
	
	if(stride == 0) {
		for(int i=0;i<m_pntBufLength;++i) {
			m_pntNormalBuf[i].set(vertN[i*3], vertN[i*3+1], vertN[i*3+2]);
			m_pntPositionBuf[i].set(vertP[i*3], vertP[i*3+1], vertP[i*3+2]);
		}
	} else {
		for(int i=0;i<m_pntBufLength;++i) {
			const float * vnml = &vertN[i*stride];
			m_pntNormalBuf[i].set(vnml[0], vnml[1], vnml[2]);
			const float * vpos = &vertP[i*stride];
			m_pntPositionBuf[i].set(vpos[0], vpos[1], vpos[2]);
		}
	}
}

void DrawPoint::drawPoints() const
{
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)m_pntNormalBuf.get() );
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_pntPositionBuf.get() );
	glDrawArrays(GL_POINTS, 0, m_pntBufLength);
}

void DrawPoint::drawWiredPoints() const
{
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)m_pntPositionBuf.get() );
	glDrawArrays(GL_POINTS, 0, m_pntBufLength);
}

}
