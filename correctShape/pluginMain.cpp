/*
 *  pluginMain.cpp
 *  
 *
 *  Created by Zhang Jian.
 *  Copyright 2010 OF3D. All rights reserved.
 *
 */

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "poseSpaceCmd.h"
#include "correctBlendShapeNode.h"
#include "recordNode.h"
 
MStatus initializePlugin( MObject obj ) 
{
	MStatus status;
	MFnPlugin plugin( obj, "OF3D | ZHANG JIAN ", "0.0.7 build Friday 12/5/12 2:15 AM", "Any" );

	status=plugin.registerCommand("calculatePoseSpace", CBPoseSpaceCmd::creator);

	if(!status)
	{
		status.perror("registerCommand");
		return status;
	}
	
	status = plugin.registerNode( "sculptToBlendTarget", CorrectBlendShapeNode::id,
						 &CorrectBlendShapeNode::creator, &CorrectBlendShapeNode::initialize );
	if (!status) {
		status.perror("registerNode");
		return status;
	}
	
	status = plugin.registerNode( "sculptSpaceRecord", PoseRecordNode::id,
						 &PoseRecordNode::creator, &PoseRecordNode::initialize );
	if (!status) {
		status.perror("registerNode");
		return status;
	}
	
	MGlobal::executeCommand ( "source correctShape.mel" );

 return MS::kSuccess;
 }
 
 MStatus uninitializePlugin( MObject obj ) 
 {
	MStatus status;
	MFnPlugin plugin( obj );

    status=plugin.deregisterCommand("calculatePoseSpace");

	if(!status)
	{
		status.perror("deregisterCommand");
		return status;
	}
	
	status = plugin.deregisterNode( CorrectBlendShapeNode::id );
	if (!status) {
		  status.perror("deregisterCommand");
		  return status;
	}
	
	status = plugin.deregisterNode( PoseRecordNode::id );
	if (!status) {
		  status.perror("deregisterCommand");
		  return status;
	}

 return MS::kSuccess;
 }


