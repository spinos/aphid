#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

#include "CinchonaNode.h"

MStatus initializePlugin( MObject obj )
{
    MString buildInfo("0.0.1 build Mon Feb 22 16:38:52 CST 2016 kdtree memory");
	MGlobal::displayInfo(MString("cinchona init plugin ") + buildInfo);
    
	MStatus   status;

	MFnPlugin plugin( obj, "OF3D | Zhang Jian", buildInfo.asChar(), "Any");

	status = plugin.registerNode( "cinchonaViz", CinchonaNode::id, 
						 &CinchonaNode::creator, &CinchonaNode::initialize,
						 MPxNode::kLocatorNode );
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
	
	status = plugin.deregisterNode( CinchonaNode::id );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}
	
	return status;
}
