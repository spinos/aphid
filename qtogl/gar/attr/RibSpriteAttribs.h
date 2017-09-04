/*
 *  RibSpriteAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_RIB_SPRITE_ATTRIBS_H
#define GAR_RIB_SPRITE_ATTRIBS_H

#include "PieceAttrib.h"

namespace aphid {
class SplineBillboard;
}

class RibSpriteAttribs : public PieceAttrib {
	
	aphid::SplineBillboard* m_billboard;
	int m_instId;
	float m_exclR;
	static int sNumInstances;
	
public:
	RibSpriteAttribs();
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(gar::SelectProfile* prof) const;
	virtual bool update();
	virtual int attribInstanceId() const;
	virtual bool isGeomLeaf() const;
	virtual void estimateExclusionRadius(float& minRadius);
	
};

#endif
