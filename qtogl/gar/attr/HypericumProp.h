/*
 *  HypericumProp.h
 *  
 *  geoms cannot change
 *
 *  Created by jian zhang on 8/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_HYPERICUM_PROP_H
#define GAR_HYPERICUM_PROP_H

#include "PieceAttrib.h"

class HypericumProp : public PieceAttrib {

	static aphid::ATriangleMesh * sMeshes[16];
	static float sExclRs[16];
	static bool sMeshesLoaded;
	static float sMeanExclR;
	
public:
	HypericumProp();

	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	virtual void estimateExclusionRadius(float& minRadius);
	
private:
	void loadMeshes();
	
};

#endif
