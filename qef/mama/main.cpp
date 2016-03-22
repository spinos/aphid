/*
 *  main.cpp
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MFnPlugin.h>
#include "qefCmd.h"

MStatus initializePlugin( MObject obj )
{ 
	MStatus   status;
	MFnPlugin plugin( obj, "Zhang Jian", "0.0.1 Tue Mar 22 00:51:14 init", "Any");

	status = plugin.registerCommand( "qef", QefCmd::creator,
												QefCmd::newSyntax);
	if (!status) {
		status.perror("register QEF Command");
		return status;
	}

	return status;
}

MStatus uninitializePlugin( MObject obj)
{
	MStatus   status;
	MFnPlugin plugin( obj );

	status = plugin.deregisterCommand( "qef" );
	if (!status) {
		status.perror("deregister QEF Cmd");
		return status;
	}

	return status;
}