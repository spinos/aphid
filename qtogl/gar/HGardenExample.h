/*
 *  HGardenExample.h
 *  garden
 *
 *  Created by jian zhang on 4/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef GAR_GARDEN_EXAMPLE_H
#define GAR_GARDEN_EXAMPLE_H

#include <h5/HBase.h>

class Vegetation;

namespace aphid {

class GardenExamp;

class HGardenExample : public HBase {

public:
	HGardenExample(const std::string & name);
	virtual ~HGardenExample();
	
	virtual char verifyType();
	char save(Vegetation * vege);
	char load(GardenExamp * vege);
	
protected:
private:
};

}

#endif