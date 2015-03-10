/*
 *  main.cpp
 *  heather
 *
 *  Created by jian zhang on 2/10/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "heatherNode.h"
#include "ExrImgData.h"
#include "ExrImgLoader.h"
#include <maya/MFnPlugin.h>
#include <CudaBase.h>

MStatus initializePlugin( MObject obj )
{ 
	MStatus   stat;
	MFnPlugin plugin( obj, "ZHANG JIAN | OF3D", "2.0 Tue Mar 10 20:54:47", "Any");

	stat = plugin.registerData("heatherData", ExrImgData::id,
								 &ExrImgData::creator);
	if (!stat) {
		stat.perror("registering rec data.");
		return stat;
	}
	
	stat = plugin.registerNode( "heatherImage", ExrImgLoader::id, 
						 &ExrImgLoader::creator, &ExrImgLoader::initialize);
	if (!stat) {
		stat.perror("registerNode");
		return stat;
	}
	
	stat = plugin.registerNode( "heatherNode", heatherNode::id, 
						 &heatherNode::creator, &heatherNode::initialize,
						 MPxNode::kLocatorNode );
	if (!stat) {
		stat.perror("registerNode");
		return stat;
	}
	
	CudaBase::SetDevice();

	// MGlobal::executeCommand ( "source cameraFrustumMenus.mel;cameraFrustumCreateMenus" );

	return stat;
}

MStatus uninitializePlugin( MObject obj)
{
	MStatus   stat;
	MFnPlugin plugin( obj );
	
	stat = plugin.deregisterData(ExrImgData::id);
	if (!stat) {
		stat.perror("deregistering exr data.");
		return stat;
	}
	
	stat = plugin.deregisterNode( ExrImgLoader::id );
	if (!stat) {
		stat.perror("deregisterNode");
		return stat;
	}

	stat = plugin.deregisterNode( heatherNode::id );
	if (!stat) {
		stat.perror("deregisterNode");
		return stat;
	}

	// MGlobal::executeCommand ( "cameraFrustumRemoveMenus" );

	return stat;
}