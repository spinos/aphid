/*
 *  SplineBillboard.h
 *
 *  billboard with spline adjust width
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_SPLINE_BILLBOARD_H
#define APH_SPLINE_BILLBOARD_H

#include "BillboardMesh.h"
#include <math/SplineMap1D.h>

namespace aphid {
    
class SplineBillboard : public BillboardMesh {
     	
	SplineMap1D m_leftSpline;
	SplineMap1D m_rightSpline;
	
public:
	SplineBillboard(float w, float h);
    virtual ~SplineBillboard();
	
	virtual void setBillboardSize(float w, float h);
	
	SplineMap1D* leftSpline();
	SplineMap1D* rightSpline();
	
	void adjustLeft();
	void adjustRight();
	
protected:
    void adjustTexcoord(bool isLeft);
	void adjustTexcoordPnt(Vector2F& texc,
			float midu, bool isLeft);
			
private:
    
};

}
#endif
