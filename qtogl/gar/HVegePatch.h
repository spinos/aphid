/*
 *  HVegePatch.h
 *  garden
 *
 *  Created by jian zhang on 5/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_HVEGE_PATH_H
#define GAR_HVEGE_PATH_H

#include <h5/HBase.h>
#include <h5/HOocArray.h>

class VegetationPatch;

namespace aphid {

class CompoundExamp;

class HVegePatch : public HBase {

public:
	HVegePatch(const std::string & name);
	virtual ~HVegePatch();
	
	virtual char verifyType();
	char save(VegetationPatch * vgp);
	char load(CompoundExamp * vgp);
	
protected:
private:
};

}

#endif