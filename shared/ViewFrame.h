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

namespace aphid {

class RectangleI {

	int m_v[4];
public:
	
	void set(int x0, int y0, int x1, int y1);
	int area() const;
	bool isLandscape() const;
	int width() const;
	int height() const;
	void split(RectangleI & r0, RectangleI & r1, float & alpha, bool alongX) const;
	
	friend std::ostream& operator<<(std::ostream &output, const RectangleI & p) {
        output << p.str();
        return output;
    }
	
	const std::string str() const;
};

class ViewFrame {

	RectangleI m_rect;
	Frustum m_frustum;
	
public:
	ViewFrame();
	virtual ~ViewFrame();

    void setRect(int x0, int y0, int x1, int y1);
    void setRect(const RectangleI & r);
    void setView(const Frustum & f);
    
    Frustum view() const;
	RectangleI rect() const;
	
	void split(ViewFrame & childLft, ViewFrame & childRgt) const;
	
	int numPixels() const;
	
	friend std::ostream& operator<<(std::ostream &output, const ViewFrame & p) {
        output << " viewframe rect " << p.rect().str();
        return output;
    }
	
};

}