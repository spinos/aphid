/*
 *  main.cpp
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
//
//	Example: geometrySurfaceConstraint
//	This example demonstrates how to use the MPxConstraint
//	and MPxConstraintCommand classes to create a 
//	geometry constraint.  This type of constraint will
//	keep the constrained object attached to the target 
//	as the target is moved.
//	The constrained object can be constrained to one of
//	mutliple targets.  You can choose to constrain to
//	the target of the highest or lowest weight.
//

/*
loadPlugin rotaConstraint;

// 1. cylinder constrained to plane
file -f -new;
polyPlane -w 1 -h 1 -sx 10 -sy 10 -ax 0 1 0 -cuv 2 -ch 1;
scale -r 15 15 15;
polyCylinder -r 1 -h 2 -sx 20 -sy 1 -sz 1 -ax 0 1 0 -rcp 0 -cuv 3 -ch 1;
select -cl;
select -r pPlane1 pCylinder1;
geometrySurfaceConstraint -weight 1;

// 2. cylinder constrained to one of two planes
// depending on plane weight
file -f -new;
polyPlane -w 1 -h 1 -sx 10 -sy 10 -ax 0 1 0 -cuv 2 -ch 1;
scale -r 10 10 10;
polyPlane -w 1 -h 1 -sx 10 -sy 10 -ax 0 1 0 -cuv 2 -ch 1;
scale -r 15 15 15;
polyCylinder -r 1 -h 2 -sx 20 -sy 1 -sz 1 -ax 0 1 0 -rcp 0 -cuv 3 -ch 1;
select -cl;
select -r pPlane1 pPlane2 pCylinder1;
geometryConstraint -weight 1.0;
// change plane weight to move constrained object
geometryConstraint -e -w 10.0 pPlane2 pCylinder1;
*/

#include <maya/MFnPlugin.h>
#include "rotaCmd.h"
#include "geometrySurfaceConstraint.h"

MStatus initializePlugin( MObject obj )
{ 
	MStatus   status;
	MFnPlugin plugin( obj, "Zhang Jian", "9.0", "Any");

	status = plugin.registerNode( "geometrySurfaceConstraint", geometrySurfaceConstraint::id, geometrySurfaceConstraint::creator,
		geometrySurfaceConstraint::initialize, MPxNode::kConstraintNode );
	if (!status) {
		status.perror("registerNode");
		return status;
	}

	status = plugin.registerConstraintCommand( "geometrySurfaceConstraint", geometrySurfaceConstraintCommand::creator );
	if (!status) {
		status.perror("registerConstraintCommand");
		return status;
	}

	return status;
}

MStatus uninitializePlugin( MObject obj)
{
	MStatus   status;
	MFnPlugin plugin( obj );

	status = plugin.deregisterNode( geometrySurfaceConstraint::id );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}

	status = plugin.deregisterConstraintCommand( "geometrySurfaceConstraint" );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}

	return status;
}