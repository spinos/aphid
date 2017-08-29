/*
 *  PotAttribs.h
 *  
 *
 *  Created by jian zhang on 8/6/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_POT_ATTRIBS_H
#define GAR_POT_ATTRIBS_H

#include "PieceAttrib.h"

class PotAttribs : public PieceAttrib {

public:
	PotAttribs();
	virtual bool isGround() const;
	virtual void getGrowthProfile(GrowthSampleProfile* prof) const;

};

#endif