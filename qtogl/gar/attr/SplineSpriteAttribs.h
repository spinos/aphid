/*
 *  SplineSpriteAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_SPLINE_SPRITE_ATTRIBS_H
#define GAR_SPLINE_SPRITE_ATTRIBS_H

#include "PieceAttrib.h"

namespace aphid {
class SplineBillboard;
}

class SplineSpriteAttribs : public PieceAttrib {
	
	aphid::SplineBillboard* m_billboard;
	int m_instId;
	
	static int sNumInstances;
	
public:
	SplineSpriteAttribs();
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(int x, float& exclR) const;
	virtual bool update();
	virtual int attribInstanceId() const;
	virtual float texcoordBlockAspectRatio() const;
	
};

#endif
