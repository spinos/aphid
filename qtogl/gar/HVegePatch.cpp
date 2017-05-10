/*
 *  HVegePatch.cpp
 *  
 *
 *  Created by jian zhang on 5/10/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "HVegePatch.h"
#include "VegetationPatch.h"
#include <math/Matrix44F.h>

namespace aphid {

HVegePatch::HVegePatch(const std::string & name) :
HBase(name) 
{}

HVegePatch::~HVegePatch() 
{}

char HVegePatch::verifyType()
{
	if(!hasNamedAttr(".bbox") ) return 0;
	if(!hasNamedAttr(".tmc") ) return 0;
	if(!hasNamedData(".tms") ) return 0;
	if(!hasNamedAttr(".npnt") ) return 0;
	if(!hasNamedData(".pntpos") ) return 0;
	if(!hasNamedData(".pntnml") ) return 0;
	if(!hasNamedData(".pntcol") ) return 0;
	return 1;
}

char HVegePatch::save(VegetationPatch * vgp)
{
	const BoundingBox & bbox = vgp->geomBox();
	if(!hasNamedAttr(".bbox"))
		addFloatAttr(".bbox", 6);
	
	writeFloatAttr(".bbox", (float *)&bbox);
	
	int tmc = vgp->getNumTms();
	if(!hasNamedAttr(".tmc"))
		addIntAttr(".tmc");
	
	writeIntAttr(".tmc", &tmc);
	
	Matrix44F * tms = new Matrix44F[tmc];
	vgp->extractTms(tms);
	
	HOocArray<hdata::TFloat, 16, 256> tmD(".tms");
	if(!hasNamedData(".tms") ) {
		tmD.createStorage(fObjectId);
	}
	
	for(int i=0;i<tmc;++i) {
		tmD.insert((char *)&tms[i]);
	}
	tmD.finishInsert();
	
	delete[] tms;
	
	int npnt = vgp->pntBufLength();
	if(!hasNamedAttr(".npnt"))
		addIntAttr(".npnt");
	
	writeIntAttr(".npnt", &npnt);
	
	if(!hasNamedData(".pntpos"))
	    addVector3Data(".pntpos", npnt);
	
	writeVector3Data(".pntpos", npnt, (Vector3F *)vgp->pntPositionBuf());
	
	if(!hasNamedData(".pntnml"))
	    addVector3Data(".pntnml", npnt);
	
	writeVector3Data(".pntnml", npnt, (Vector3F *)vgp->pntNormalBuf());
	
	if(!hasNamedData(".pntcol"))
	    addVector3Data(".pntcol", npnt);
	
	writeVector3Data(".pntcol", npnt, (Vector3F *)vgp->pntColorBuf());
	
	return 1;
}

}