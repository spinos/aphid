/*
 *  HGardenExample.cpp
 *  garden
 *
 *  Created by jian zhang on 4/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "HGardenExample.h"
#include "Vegetation.h"
#include "VegetationPatch.h"
#include <math/Matrix44F.h>
#include <geom/ATriangleMesh.h>
#include <h5/HTriangleMesh.h>
#include "HVegePatch.h"
#include <sstream>

namespace aphid {

HGardenExample::HGardenExample(const std::string & name) :
HBase(name) 
{}

HGardenExample::~HGardenExample() 
{}
	
char HGardenExample::verifyType()
{
	if(!hasNamedAttr(".bbox") ) return 0;
	if(!hasNamedAttr(".xmpc") ) return 0;
	if(!hasNamedAttr(".geoc") ) return 0;
	return 1;
}

char HGardenExample::save(Vegetation * vege)
{
	std::cout<<" HGardenExample save "<<fObjectPath;
	
	const BoundingBox & bbox = vege->bbox();
	if(!hasNamedAttr(".bbox"))
		addFloatAttr(".bbox", 6);
	
	writeFloatAttr(".bbox", (float *)&bbox);
	
	int nxmp = vege->numPatches();
	if(!hasNamedAttr(".xmpc"))
		addIntAttr(".xmpc");
	
	writeIntAttr(".xmpc", &nxmp);
	
	std::cout<<"\n n example "<<nxmp;
	
	std::stringstream sst;
	for(int i=0;i<nxmp;++i) {
		sst.str("vgp_");
		sst<<i;
		VegetationPatch * vgp = vege->patch(i);
		std::string vgpPathName = childPath(sst.str().c_str() );
		HVegePatch chd(vgpPathName);
		chd.save(vgp);
		chd.close();
	}
	
	int ngeo = vege->numCachedGeoms();
	if(!hasNamedAttr(".geoc"))
		addIntAttr(".geoc");
	
	writeIntAttr(".geoc", &ngeo);
	
	std::cout<<"\n n geom "<<ngeo;
	
	std::string mshName, mshPathName;
	ATriangleMesh * mshVal = NULL;
	vege->geomBegin(mshName, mshVal);
	while(mshVal) {
		mshPathName = childPath(mshName.c_str() );
		HTriangleMesh chd(mshPathName);
		chd.save(mshVal);
		chd.close();
		vege->geomNext(mshName, mshVal);
	}
	
	std::cout.flush();
	return 1;
}

}