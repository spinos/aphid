/*
 *  SimpleTrunkProp.h
 *  
 *  geoms cannot change
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SIMPLE_TRUNK_PROP_H
#define GAR_SIMPLE_TRUNK_PROP_H

#include "PieceAttrib.h"

class SimpleTrunkProp : public PieceAttrib {

	static aphid::ATriangleMesh * sMesh;
	static float sExclR;
	static bool sMeshLoaded;
	
public:
	SimpleTrunkProp();

	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	virtual void estimateExclusionRadius(float& minRadius);
	
private:
	void loadMesh();
	
};

#endif
