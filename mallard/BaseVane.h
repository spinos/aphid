/*
 *  BaseVane.h
 *  mallard
 *
 *  Created by jian zhang on 12/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BezierCurve.h>
class BaseVane {
public:
    BaseVane();
    virtual ~BaseVane();
	
	virtual void setU(float u);
    
    virtual void create(unsigned gridU, unsigned gridV);
    BezierCurve * profile(unsigned idx) const;
    void computeKnots();
    void pointOnVane(float v, Vector3F & dst);
	Vector3F * railCV(unsigned u, unsigned v);
	unsigned gridU() const;
	unsigned gridV() const;
	
	BezierCurve * profile();
	BezierCurve * rails();
private:
    BezierCurve m_profile;
    BezierCurve * m_rails;
    unsigned m_gridU, m_gridV;
};
