/*
 *  HesperisInterface.cpp
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisInterface.h"
#include "HesperisFile.h"
#include <CurveGroup.h>
#include <GeometryArray.h>
#include <KdTreeDrawer.h>
#include <APointCloud.h>

std::string HesperisInterface::FileName("unknown");
HesperisInterface::HesperisInterface() {}
HesperisInterface::~HesperisInterface() {}

bool HesperisInterface::CheckFileExists()
{
	if(BaseFile::InvalidFilename(FileName)) 
		return false;
		
	if(!BaseFile::FileExists(FileName)) {
		FileName = "unknown";
		return false;
	}
	
	return true;
}

bool HesperisInterface::ReadCurveData(CurveGroup * data)
{
	if(!CheckFileExists()) return false;
	
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RCurve);
	hes.addCurve("curves", data);
	if(!hes.open(FileName)) return false;
	hes.close();
	
	return true;
}

bool HesperisInterface::ReadTriangleData(GeometryArray * data)
{
	if(!CheckFileExists()) return false;
	
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTri);
	if(!hes.open(FileName)) return false;
	hes.close();
	
	hes.extractTriangleMeshes(data);
	
	return data->numGeometries() > 0;
}

bool HesperisInterface::ReadTetrahedronData(GeometryArray * data)
{
	if(!CheckFileExists()) return false;
	
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTetra);
	if(!hes.open(FileName)) return false;
	hes.close();
	
	hes.extractTetrahedronMeshes(data);
	
	return data->numGeometries() > 0;
}
//:~