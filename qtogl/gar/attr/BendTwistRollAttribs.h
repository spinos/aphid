/*
 *  BendTwistRollAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_BEND_TWIST_ROLL_ATTRIBS_H
#define GAR_BEND_TWIST_ROLL_ATTRIBS_H

#include "PieceAttrib.h"

class BendTwistRollAttribs : public PieceAttrib {

	aphid::ATriangleMesh* m_inGeom;
	aphid::ATriangleMesh* m_outGeom[64];
	int m_instId;
	static int sNumInstances;
	
public:
	BendTwistRollAttribs();
	
	void setInputGeom(aphid::ATriangleMesh* x);
	
	virtual bool hasGeom() const;
	virtual int numGeomVariations() const;
	virtual aphid::ATriangleMesh* selectGeom(int x, float& exclR) const;
	virtual bool update();
/// multi instance of different settings
	virtual int attribInstanceId() const;
/// recv input geom
	virtual void connectTo(PieceAttrib* another);
};

#endif