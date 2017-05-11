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
#include <CompoundExamp.h>
#include <GardenExamp.h>
#include "HVegePatch.h"
#include <boost/format.hpp>
#include <iomanip>

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
	
	std::string vgpName;
	for(int i=0;i<nxmp;++i) {
		vgpName = str(boost::format("vgp_%1%") % boost::io::group(std::setw(3), std::setfill('0'), i) );
		
		VegetationPatch * vgp = vege->patch(i);
		std::string vgpPathName = childPath(vgpName.c_str() );
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
	std::cout<<"\n done saving garden example ";
	std::cout.flush();
	return 1;
}

char HGardenExample::load(GardenExamp * vege)
{
	std::cout<<" HGardenExample load "<<fObjectPath;
	
	BoundingBox bbox;
	readFloatAttr(".bbox", (float *)&bbox);
	vege->setGeomBox2(bbox);
	
	int nxmp = 0;
	readIntAttr(".xmpc", &nxmp);
	std::cout<<"\n n example "<<nxmp;
	
	std::vector<std::string> exmpNames;
	lsTypedChild<HVegePatch > (exmpNames);
	
	std::vector<std::string>::const_iterator it = exmpNames.begin();
	for(;it!=exmpNames.end();++it) {
		CompoundExamp * vgp = new CompoundExamp;
		HVegePatch chd(*it);
		chd.load(vgp);
		chd.close();
		vege->addAExample(vgp);
	}
	
	return 1;
}

}