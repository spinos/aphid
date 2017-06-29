/*
 *  GardenWorks.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 5/2/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "GardenWorks.h"
#include <h5/HObject.h>
#include <h5/HDocument.h>
#include <h5/HTriangleMesh.h>
#include <HGardenExample.h>
#include <VegExampleNode.h>
#include <geom/ATriangleMesh.h>
#include <mama/AHelper.h>
#include <mama/MeshHelper.h>
#include <mama/ConnectionHelper.h>

using namespace aphid;

GardenWorks::GardenWorks()
{}

GardenWorks::~GardenWorks()
{}

MStatus GardenWorks::importGardenFile(const char * fileName)
{
	HDocument h5doc;
	if(!h5doc.open(fileName, HDocument::oReadOnly)) {
		AHelper::Info<const char * > ("GardenWorks error cannot open gde file ", fileName);
        return MS::kFailure;
	}
	
	HObject::FileIO = h5doc;
	
	MStatus stat = MS::kSuccess;
	
	HBase gr("/");
	
	std::vector<std::string > gdeNames;
	gr.lsTypedChild<HGardenExample > (gdeNames);
	
	if(gdeNames.size() > 0) {
		stat = doImport(gdeNames[0]);
        
	} else {
		AHelper::Info<const char * > ("GardenWorks error cannot find example in gde file ", fileName);
        stat = MS::kFailure;
	}
	
	gr.close();
	
	h5doc.close();
	
	AHelper::Info<const char * > ("GardenWorks finished importing gde file ", fileName);
	
	return stat;
}

MStatus GardenWorks::doImport(const std::string & gdeName)
{
	AHelper::Info<std::string > ("GardenWorks import example ", gdeName);
	MStatus stat = MS::kSuccess;
	
	HGardenExample gd(gdeName);
	
	const MString sgde(gd.lastName().c_str() );
	MObject od = AHelper::CreateTransform(sgde);
	if(od.isNull() ) {
		AHelper::Info<MString > ("GardenWorks error cannot create transform ", sgde);
        gd.close();
		return MS::kFailure;
	}
	
	MObject gde = importExample(&gd, &od, &stat);
	if(!gde.isNull() ) importMesh(&gd, &od, &gde);
	
	gd.close();
	
	return stat;
}

MObject GardenWorks::importExample(aphid::HGardenExample * grp,
			MObject * parent, MStatus * stat)
{
	const MString vizName = MFnDependencyNode(*parent).name() + "Shape";
	MObject oviz = AHelper::CreateShapeNode("gardenExampleViz", vizName, *parent);
	if(oviz.isNull() ) {
		AHelper::Info<std::string > ("GardenWorks error cannot create shape ", grp->lastName() );
		*stat = MS::kFailure;
		return oviz;
	}
	
	VegExampleNode * viz = (VegExampleNode *)MFnDependencyNode(oviz).userNode();
	grp->load(viz);
	viz->saveInternal();
	viz->buildAllExmpVoxel();
	*stat = MS::kSuccess;
	return oviz;
}

MStatus GardenWorks::importMesh(HGardenExample * grp,
							MObject * parent, MObject * gde)
{
	MStatus stat = MS::kSuccess;
	std::vector<std::string > mshNames;
	grp->lsTypedChild<HTriangleMesh > (mshNames);
	
	const int n = mshNames.size();
	for(int i=0;i<n;++i) {
		std::cout<<" GardenWorks import geom "<< mshNames[i];
		
		HTriangleMesh gm(mshNames[i]);
		
		ATriangleMesh td;
		gm.load(&td);
		gm.loadTriangleTexcoord(&td);
		
		const MString smsh(gm.lastName().c_str() );
		MObject om = AHelper::CreateTransform(smsh, *parent);
		
		ConnectionHelper::ConnectToArray(om, "matrix", 
										*gde, "instanceSpace",
										-1);
		
		MeshHelper::CreateProfile prof;
		prof._hasUV = true;
		MObject tg = MeshHelper::CreateMesh(td, om, &prof);
		
		if(tg != MObject::kNullObj) {
			MFnDependencyNode dpf(tg);
			dpf.setName(smsh + "Shape");
		}
		
		gm.close();
		
	} 
	
	if(n < 1) {
		AHelper::Info<const char * > ("GardenWorks error cannot find geom in example ", "");
        stat = MS::kFailure;
	}
	
	return stat;
}
