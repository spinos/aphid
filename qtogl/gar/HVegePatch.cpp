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
#include <CompoundExamp.h>

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
	if(!hasNamedAttr(".dopnv") ) return 0;
	if(!hasNamedData(".doppos") ) return 0;
	if(!hasNamedData(".dopnml") ) return 0;
	if(!hasNamedAttr(".grdnv") ) return 0;
	if(!hasNamedData(".grdpos") ) return 0;
	if(!hasNamedData(".grdnml") ) return 0;
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
	
	int * geoid = new int[tmc];
	vgp->extractGeomIds(geoid);
	
	if(!hasNamedData(".geoid"))
	    addIntData(".geoid", tmc);
	
	writeIntData(".geoid", tmc, geoid);
	
	delete[] geoid;
	
	int npnt = vgp->pntBufLength();
	addVertexBlock2(".npnt", ".pntpos", ".pntnml", ".pntcol",
					&npnt, (Vector3F *)vgp->pntPositionBuf(),
					(Vector3F *)vgp->pntNormalBuf(),
					(Vector3F *)vgp->pntColorBuf() );
	
	int dopnv = vgp->dopBufLength();
	addVertexBlock(".dopnv", ".doppos", ".dopnml",
					&dopnv, (Vector3F *)vgp->dopPositionBuf(),
					(Vector3F *)vgp->dopNormalBuf());
	
	int grdnv = vgp->grdBufLength();
	addVertexBlock(".grdnv", ".grdpos", ".grdnml",
					&grdnv, (Vector3F *)vgp->grdPositionBuf(),
					(Vector3F *)vgp->grdNormalBuf());
	
	return 1;
}

char HVegePatch::load(CompoundExamp * vgp)
{
	BoundingBox bbox;
	readFloatAttr(".bbox", (float *)&bbox);
	vgp->setGeomBox2(bbox);
	
	int tmc = 0;
	readIntAttr(".tmc", &tmc);
	
	Matrix44F * tms = new Matrix44F[tmc];
	HOocArray<hdata::TFloat, 16, 256> tmD(".tms");
	tmD.openStorage(fObjectId);
	const int & ntm = tmD.numCols();
	for(int i=0; i<ntm; ++i) {
	    tmD.readColumn((char *)&tms[i], i);
	}
	
	int * geoid = new int[tmc];
	readIntData(".geoid", tmc, geoid);
	
	for(int i=0; i<ntm; ++i) {
		vgp->addInstance(tms[i], geoid[i]);
	}
	
	delete[] geoid;
	delete[] tms;
	
	int npnt = 0;
	readIntAttr(".npnt", &npnt);
	vgp->setPointDrawBufLen(npnt);
	readVector3Data(".pntpos", npnt, vgp->pntPositionR() );
	readVector3Data(".pntnml", npnt, vgp->pntNormalR() );
	readVector3Data(".pntcol", npnt, vgp->pntColorR() );
	
	int dopnv = 0;
	readIntAttr(".dopnv", &dopnv);
	vgp->setDopDrawBufLen(dopnv);
	readVector3Data(".doppos", dopnv, (Vector3F *)vgp->dopRefPositionR() );
	readVector3Data(".dopnml", dopnv, (Vector3F *)vgp->dopNormalR() );
	vgp->resizeDopPoints(Vector3F(1.f, 1.f, 1.f) );
	
	int grdnv = 0;
	readIntAttr(".grdnv", &grdnv);
	vgp->setGrdDrawBufLen(grdnv);
	readVector3Data(".grdpos", grdnv, (Vector3F *)vgp->grdPositionR() );
	readVector3Data(".grdnml", grdnv, (Vector3F *)vgp->grdNormalR() );
	
	return 1;
}

}