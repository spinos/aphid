/*
 *  GardenWorks.h
 *  proxyPaint
 *
 *  Created by jian zhang on 5/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef PXPNT_GARDEN_WORKS_H
#define PXPNT_GARDEN_WORKS_H

#include <maya/MStatus.h>
#include <maya/MObject.h>
#include <string>

namespace aphid {

class HGardenExample;

}

class GardenWorks {

public:
	GardenWorks();
	virtual ~GardenWorks();
	
protected:
	MStatus importGardenFile(const char * fileName);

private:
	MStatus doImport(const std::string & gdeName);
	MObject importExample(aphid::HGardenExample * grp,
			MObject * parent, MStatus * stat);
	MStatus importMesh(aphid::HGardenExample * grp,
			MObject * parent, MObject * gde);
	
};
#endif
