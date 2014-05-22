/*
 *  pluginMain.cpp
 *
 *  2.4.5 - 06.05.08
 *
 */

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

#include "SolverNode.h"
#include "RigidBodyTransform.h"
#include "ConditionNode.h"

MStatus initializePlugin( MObject obj )
{
    MString buildInfo = MString("0.0.1 build Thu May 22 00:47:01 CST 2014 init");

	MStatus   status;

	MFnPlugin plugin( obj, "Zhang Jian", buildInfo.asChar(), "Any");
	

	status = plugin.registerNode( "caterpillarSolver", caterpillar::SolverNode::id, 
						 &caterpillar::SolverNode::creator, &caterpillar::SolverNode::initialize,
						 MPxNode::kLocatorNode );
	if (!status) {
		status.perror("registerNode");
		return status;
	}
	
	status = plugin.registerNode( "caterpillarCondition", caterpillar::ConditionNode::id, 
						 &caterpillar::ConditionNode::creator, &caterpillar::ConditionNode::initialize,
						 MPxNode::kLocatorNode );
	if (!status) {
		status.perror("registerNode");
		return status;
	}
	
	status = plugin.registerTransform( "caterpillarRigidBody", 
									caterpillar::RigidBodyTransformNode::id, 
									&caterpillar::RigidBodyTransformNode::creator, 
									&caterpillar::RigidBodyTransformNode::initialize,
									&caterpillar::RigidBodyTransformMatrix::creator,
									caterpillar::RigidBodyTransformMatrix::id);
	if (!status) {
		status.perror("registerNode");
		return status;
	}
	
	return status;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus   status;
	MFnPlugin plugin( obj );
	
	status = plugin.deregisterNode(caterpillar::SolverNode::id );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}
	
	status = plugin.deregisterNode(caterpillar::ConditionNode::id );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}
	
	status = plugin.deregisterNode( caterpillar::RigidBodyTransformNode::id );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}
	
	return status;
}
