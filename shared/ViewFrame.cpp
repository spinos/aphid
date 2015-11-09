/*
 *  ViewFrame.cpp
 *  
 *
 *  Created by jian zhang on 11/5/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "ViewFrame.h"

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


ViewFrame::ViewFrame() {}
ViewFrame::~ViewFrame() {}

void ViewFrame::setRect(int x0, int y0, int x1, int y1)
{ m_rect.set(x0, y0, x1, y1); }

void ViewFrame::setView(const Frustum & f)
{ m_frustum = f; }

Frustum ViewFrame::view() const
{ return m_frustum; }