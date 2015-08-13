#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "HesperisCmd.h"
#include "AdaptiveFieldDeformer.h"
#include "BoundTranslateNode.h"

static const MString Verinfo("1.3.7 Fri Aug 14 04:17:29 CST 2015 bounding box center as translate");
MStatus initializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj, "Zhang Jian", Verinfo.asChar(), "Any");

		status = plugin.registerNode( "hesperisDeformer", AdaptiveFieldDeformer::id,
						 &AdaptiveFieldDeformer::creator, &AdaptiveFieldDeformer::initialize,
						 MPxNode::kDeformerNode );
		if (!status) {
			status.perror("registerDeformer");
			return status;
		}
        
        status = plugin.registerNode( "hesperisTranslateNode", BoundTranslateNode::id, BoundTranslateNode::creator,
                                        BoundTranslateNode::initialize );
        if (!status) {
            status.perror("registerNode");
            return status;
        }
		
        status = plugin.registerCommand("hesperis",
                                        HesperisCmd::creator, HesperisCmd::newSyntax);
        if (!status) {
                status.perror("registerCommand");
                return status;
        }
//  connectAttr -f pSphere1.boundingBoxMin hesperisTranslateNode1.inBoundingBoxMin;
//  connectAttr -f pSphere1.boundingBoxMax hesperisTranslateNode1.inBoundingBoxMax;
        MGlobal::displayInfo(MString("hesperis load plug-in version ") + Verinfo);
        return status;
}

MStatus uninitializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj);
        
        AdaptiveFieldDeformer::CloseAllFiles();
		
		status = plugin.deregisterNode( AdaptiveFieldDeformer::id );
		if (!status) {
			  status.perror("deregisterDeformer");
			  return status;
		}
        
        status = plugin.deregisterNode( BoundTranslateNode::id );
        if (!status) {
            status.perror("deregisterNode");
            return status;
        }

        status = plugin.deregisterCommand("hesperis");
        if (!status) {
                status.perror("deregisterCommand");
                return status;
        }
        
        MGlobal::displayInfo(MString("hesperis unload plug-in version ") + Verinfo);
        
        return status;
}

