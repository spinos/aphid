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

ViewFrame::ViewFrame() {}
ViewFrame::~ViewFrame() {}

void ViewFrame::setRect(int x0, int y0, int x1, int y1)
{ m_rect.set(x0, y0, x1, y1); }

void ViewFrame::setRect(const RectangleI & r)
{ m_rect = r; }

void ViewFrame::setView(const cvx::Frustum & f)
{ m_frustum = f; }

cvx::Frustum ViewFrame::view() const
{ return m_frustum; }

RectangleI ViewFrame::rect() const
{ return m_rect; }

void ViewFrame::split(ViewFrame & childLft, ViewFrame & childRgt) const
{ 
	RectangleI r0, r1;
	const bool alongX = rect().isLandscape();
	float alpha;
	rect().split(r0, r1, alpha, alongX);
	cvx::Frustum f0, f1;
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