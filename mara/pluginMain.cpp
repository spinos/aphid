/*
 *  Mallard Maya Plug-in
 *  pluginMain.cpp
 *
 *  10/14/2013
 *
 */
#ifdef WIN32
#include "../license/TimeCheck.h"
#endif
#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>

#include "MaraNode.h"

MStatus initializePlugin( MObject obj )
{ 
	MGlobal::displayInfo("Mallard initialize plug-in");
#ifdef WIN32
    TimeCheck tc;
    if(!tc.noLaterThan("20131231")) {
        MGlobal::executeCommand("confirmDialog -title \"Mallard License Error\" -message \"Your license to use Proxy Paint has expired! Please contact Original Force to renew your license.\"");
        return MS::kFailure;
    }
#endif
    MString buildInfo = MString("build 0.0.1 Wed 1:22 AM 11/26/2012");
	MStatus   status;

	MFnPlugin plugin( obj, "OF3D | Zhang Jian", buildInfo.asChar(), "Any");
	

	status = plugin.registerNode( "MallardViz", MallardViz::id, 
						 &MallardViz::creator, &MallardViz::initialize,
						 MPxNode::kLocatorNode );
	if (!status) {
		status.perror("registerNode");
		return status;
	}
											
	//MGlobal::executeCommand("source proxyPaintMenus.mel; proxyPaintMakeMenus;");

	return status;
}

MStatus uninitializePlugin( MObject obj )
{
	MStatus   status;
	MFnPlugin plugin( obj );
	
	status = plugin.deregisterNode( MallardViz::id );
	if (!status) {
		status.perror("deregisterNode");
		return status;
	}
		
	//MGlobal::executeCommand("proxyPaintRemoveMenus;");

	return status;
}
