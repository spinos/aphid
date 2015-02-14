/*
 *  main.cpp
 *  heather
 *
 *  Created by jian zhang on 2/10/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include "heatherNode.h"
#include <maya/MFnPlugin.h>

MStatus initializePlugin( MObject obj )
{ 
	MStatus   stat;
	MFnPlugin plugin( obj, "ZHANG JIAN | OF3D", "1.1 Sat Feb 14 18:27:07 CST 2015", "Any");

	stat = plugin.registerNode( "heatherNode", heatherNode::id, 
						 &heatherNode::creator, &heatherNode::initialize,
						 MPxNode::kLocatorNode );
	if (!stat) {
		stat.perror("registerNode");
		return stat;
	}

	// MGlobal::executeCommand ( "source cameraFrustumMenus.mel;cameraFrustumCreateMenus" );

	return stat;
}

MStatus uninitializePlugin( MObject obj)
{
	MStatus   stat;
	MFnPlugin plugin( obj );

	stat = plugin.deregisterNode( heatherNode::id );
	if (!stat) {
		stat.perror("deregisterNode");
		return stat;
	}

	// MGlobal::executeCommand ( "cameraFrustumRemoveMenus" );

	return stat;
}