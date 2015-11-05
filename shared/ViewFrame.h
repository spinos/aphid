/*
 *  ViewFrame.h
 *  
 *
 *  Created by jian zhang on 11/5/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <ConvexShape.h>

class RectangleI {

	int m_v[4];
public:
	
	void set(int x0, int y0, int x1, int y1);
	int area() const;
	bool isLandscape() const;
	int width() const;
	int height() const;
};

class ViewFrame {

	RectangleI m_rect;
	Frustum m_frustum;
	
public:
	ViewFrame();
	virtual ~ViewFrame();

    void setRect(int x0, int y0, int x1, int y1);
    void setView(const Frustum & f);
};