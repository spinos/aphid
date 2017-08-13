/*
 *  SplineCylinderAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SPLINE_CYLINDER_ATTRIBS_H
#define GAR_SPLINE_CYLINDER_ATTRIBS_H

#include "PieceAttrib.h"

namespace aphid {
class SplineCylinder;
}

class SplineCylinderAttribs : public PieceAttrib {
	
	aphid::SplineCylinder* m_cylinder;
	int m_instId;
	
	static int sNumInstances;
	
public:
	SplineCylinderAttribs();
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(int x, float& exclR) const;
	virtual bool update();
	virtual int attribInstanceId() const;
	virtual float texcoordBlockAspectRatio() const;
	
};

#endif
