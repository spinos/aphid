/*
 *  ViewFrame.cpp
 *  
 *
 *  Created by jian zhang on 11/5/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ViewFrame.h"

namespace aphid {

void RectangleI::set(int x0, int y0, int x1, int y1)
{
	m_v[0] = x0; m_v[1] = y0;
	m_v[2] = x1; m_v[3] = y1;
}

int RectangleI::width() const
{ return m_v[2] - m_v[0] + 1; }

int RectangleI::height() const
{ return m_v[3] - m_v[1] + 1; }

int RectangleI::area() const
{ return width() * height(); }

bool RectangleI::isLandscape() const
{ return width() >= height(); }

const std::string RectangleI::str() const
{
	std::stringstream sst;
	sst<<"(("<<m_v[0]<<","<<m_v[1]<<"),("<<m_v[2]<<","<<m_v[3]<<"))";
	return sst.str();
}

void RectangleI::split(RectangleI & r0, RectangleI & r1, float & alpha, bool alongX) const
{
	int s;
	if(alongX) {
		s = width() / 2;
		if(s & 31) s = (s / 32 + 1)<<5;
		alpha = (float)s/(float)width();
		s--;
		r0.set(m_v[0],       m_v[1], m_v[0] + s, m_v[3]);
		r1.set(m_v[0] + s+1, m_v[1], m_v[2],     m_v[3]);
	}
	else {
		s = height() / 2;
		if(s & 31) s = (s / 32 + 1)<<5;
		alpha = (float)s/(float)height();
		s--;
		r0.set(m_v[0], m_v[1],       m_v[2], m_v[1] + s);
		r1.set(m_v[0], m_v[1] + s+1, m_v[2],     m_v[3]);
	}
}


ViewFrame::ViewFrame() {}
ViewFrame::~ViewFrame() {}

void ViewFrame::setRect(int x0, int y0, int x1, int y1)
{ m_rect.set(x0, y0, x1, y1); }

void ViewFrame::setRect(const RectangleI & r)
{ m_rect = r; }

void ViewFrame::setView(const Frustum & f)
{ m_frustum = f; }

Frustum ViewFrame::view() const
{ return m_frustum; }

RectangleI ViewFrame::rect() const
{ return m_rect; }

void ViewFrame::split(ViewFrame & childLft, ViewFrame & childRgt) const
{ 
	RectangleI r0, r1;
	const bool alongX = rect().isLandscape();
	float alpha;
	rect().split(r0, r1, alpha, alongX);
	Frustum f0, f1;
	m_frustum.split(f0, f1, alpha, alongX);
	childLft.setRect(r0);
	childRgt.setRect(r1);
	childLft.setView(f0);
	childRgt.setView(f1);
}

int ViewFrame::numPixels() const
{ return m_rect.area(); }

}
//:~