/*
 *  pluginMain.cpp
 *
 *  2.4.5 - 06.05.08
 *
 */

#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

#include "bciosVizNode.h"

// The initializePlugin method is called by Maya when the
// plugin is loaded. It registers the hwbruiseMapShader node
// which provides Maya with the creator and initialize methods to be
// called when a hwbruiseMapShader is created.
//
MStatus initializePlugin( MObject obj )
{ 
    
	MStatus   status;
	MFnPlugin plugin( obj, "OF3D | Zhang Jian", "0.0.3 - Tuesday September 11 2011", "Any");
											
	status = plugin.registerNode( "barycentricInterpolationViz", BCIViz::id, 
						 &BCIViz::creator, &BCIViz::initialize,
						 MPxNode::kLocatorNode );
	if (!status) {
		status.perror("registerNode");
		return status;
	}

	return status;
}

// The unitializePlugin is called when Maya needs to unload the plugin.
// It basically does the opposite of initialize by calling
// the deregisterNode to remove it.
//
MStatus uninitializePlugin( MObject obj )
{
	MStatus   status;
	MFnPlugin plugin( obj );
	
	status = plugin.deregisterNode( BCIViz::id );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}

	return status;
}
