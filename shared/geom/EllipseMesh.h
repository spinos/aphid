/*
 *  EllipseMesh.h
 * 
 *  major axis along +y
 *  m # v segments n # segments on one side
 *  2n - 1 profiles
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_ELLIPSE_MESH_H
#define APH_ELLIPSE_MESH_H

#include "LoftMesh.h"
#include <math/SplineMap1D.h>

namespace aphid {

class EllipseMesh : public LoftMesh {
	
	SplineMap1D m_leftSpline;
	SplineMap1D m_rightSpline;
/// vertical
	SplineMap1D m_heightSpline;
/// horizontal
	SplineMap1D m_veinSpline;
/// vertical vein scale
	SplineMap1D m_veinVarySpline;
		
public:
	EllipseMesh();
	virtual ~EllipseMesh();

	virtual void createEllipse(const float& width, const float& height,
					const int& m, const int& n);
	
	SplineMap1D* leftSpline();
	SplineMap1D* rightSpline();
	SplineMap1D* heightSpline();
	SplineMap1D* veinSpline();
	SplineMap1D* veinVarySpline();
	
protected:
	
private:
};

}
#endif
