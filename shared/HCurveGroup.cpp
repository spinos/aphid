/*
 *  HCurveGroup.cpp
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HCurveGroup.h"
#include <AllHdf.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <CurveGroup.h>
HCurveGroup::HCurveGroup(const std::string & path) : HBase(path) 
{
}

HCurveGroup::~HCurveGroup() {}

char HCurveGroup::verifyType()
{
	if(!hasNamedAttr(".nv"))
		return 0;

	if(!hasNamedAttr(".nc"))
		return 0;
	
	return 1;
}

char HCurveGroup::save(CurveGroup * curve)
{
	curve->verbose();
	
	int nv = curve->numPoints();
	if(!hasNamedAttr(".nv"))
		addIntAttr(".nv");
	
	writeIntAttr(".nv", &nv);
	
	int nc = curve->numCurves();
	if(!hasNamedAttr(".nc"))
		addIntAttr(".nc");
	
	writeIntAttr(".nc", &nc);
	
	if(!hasNamedData(".p"))
	    addVector3Data(".p", nv);
	
	writeVector3Data(".p", nv, curve->points());
		
	if(!hasNamedData(".cc"))
	    addIntData(".cc", nc);
	
	writeIntData(".cc", nc, (int *)curve->counts());

	return 1;
}

char HCurveGroup::load(CurveGroup * curve)
{
	if(!verifyType()) return false;
	int numPoints = 4;
	
	readIntAttr(".nv", &numPoints);
	
	int numCurves = 1;
	
	readIntAttr(".nc", &numCurves);
	
	curve->create(numCurves, numPoints);
	
	readVector3Data(".p", numPoints, curve->points());
	readIntData(".cc", numCurves, curve->counts());
	
	curve->verbose();
	
	return 1;
}
//:~