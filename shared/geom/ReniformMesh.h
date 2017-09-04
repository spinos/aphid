/*
 *  ReniformMesh.h
 * 
 *  major axis along +y
 *  2 + 13 profiles
 *
 *  Created by jian zhang on 8/20/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_RENIFORM_MESH_H
#define APH_RENIFORM_MESH_H

#include "LoftMesh.h"
#include <math/SplineMap1D.h>

namespace aphid {

struct ReniformMeshProfile {
	float _kidneyAngle, _radius, _stalkWidth, _stalkHeight, _midribHeight, _ascendAngle; 
	int _stalkSegments, _veinSegments;
};

class ReniformMesh : public LoftMesh {

/// radius variation	
	SplineMap1D m_leftSideSpline;
	SplineMap1D m_rightSideSpline;
/// radial
	SplineMap1D m_veinSpline;
/// along fan vein scale
	SplineMap1D m_veinVarySpline;
		
public:
	ReniformMesh();
	virtual ~ReniformMesh();

	virtual void createReniform(const ReniformMeshProfile& prof);
	
	SplineMap1D* leftSideSpline();
	SplineMap1D* rightSideSpline();
	SplineMap1D* veinSpline();
	SplineMap1D* veinVarySpline();
	
protected:
	
private:
};

}
#endif
