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
#include <BoundingRectangle.h>

namespace aphid {

class ViewFrame {

	RectangleI m_rect;
	cvx::Frustum m_frustum;
	
public:
	ViewFrame();
	virtual ~ViewFrame();

    void setRect(int x0, int y0, int x1, int y1);
    void setRect(const RectangleI & r);
    void setView(const cvx::Frustum & f);
    
    cvx::Frustum view() const;
	RectangleI rect() const;
	
	void split(ViewFrame & childLft, ViewFrame & childRgt) const;
	
	int numPixels() const;
	
	friend std::ostream& operator<<(std::ostream &output, const ViewFrame & p) {
        output << " viewframe rect " << p.rect().str();
        return output;
    }
	
};

}