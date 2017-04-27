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

namespace aphid {

HGardenExample::HGardenExample(const std::string & name) :
HBase(name) 
{}

HGardenExample::~HGardenExample() 
{}
	
char HGardenExample::verifyType()
{
	if(!hasNamedAttr(".xmpc") ) return 0;
	if(!hasNamedData(".tmblk") ) return 0;
	if(!hasNamedData(".tms") ) return 0;
	if(!hasNamedAttr(".geoc") ) return 0;
	return 1;
}

char HGardenExample::save(Vegetation * vege)
{
	std::cout<<" HGardenExample save "<<fObjectPath;
	
	int nxmp = vege->numPatches();
	if(!hasNamedAttr(".xmpc"))
		addIntAttr(".xmpc");
	
	writeIntAttr(".xmpc", &nxmp);
	
	std::cout<<"\n n example "<<nxmp;
	
	int * blks = new int[nxmp+1];
	int cc = 0;
	for(int i=0;i<nxmp;++i) {
		blks[i] = cc;
		VegetationPatch * vgp = vege->patch(i);
		cc += vgp->getNumTms();
	}
	blks[nxmp] = cc;
	
	std::cout<<"\n n instance "<<cc;
	
	if(!hasNamedData(".tmblk"))
	    addIntData(".tmblk", nxmp+1);
	
	writeIntData(".tmblk", nxmp+1, (int *)blks);
	
	Matrix44F * tms = new Matrix44F[cc];
	for(int i=0;i<nxmp;++i) {
		VegetationPatch * vgp = vege->patch(i);
		vgp->extractTms(&tms[blks[i]]);
	}
	
	HOocArray<hdata::TFloat, 16, 256> tmD(".tms");
	if(!hasNamedData(".tms") ) {
		tmD.createStorage(fObjectId);
	}
	
	for(int i=0;i<cc;++i) {
		tmD.insert((char *)&tms[i]);
	}
	tmD.finishInsert();
	
	delete[] tms;
	delete[] blks;
	
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