/*
 *  pluginMain.cpp
 *  conformare
 *
 *  Created by jian zhang on 12/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "calcHarmonicCoordCmd.h"

MStatus initializePlugin( MObject obj )
{
	MStatus   status;
    MFnPlugin plugin( obj, "OF3D | Zhang Jian", "0.0.2 Wednesday April 10 2013", "Any");

    status = plugin.registerCommand( "calcHarmCoord", HarmonicCoordCmd::creator );
	if (!status) {
		status.perror("registerCommand");
		return status;
	}
	
	return status;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus   status;
    MFnPlugin plugin( obj );

    status = plugin.deregisterCommand( "calcHarmCoord" );
	if (!status) {
		status.perror("deregisterCommand");
	}
	
	return status;
}