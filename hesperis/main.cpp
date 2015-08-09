#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "HesperisCmd.h"
#include "AdaptiveFieldDeformer.h"

MStatus initializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj, "Zhang Jian", "1.3.4 Sat Jul 18 18:27:54 CST 2015 grow mesh in world space", "Any");

		status = plugin.registerNode( "hesperisDeformer", AdaptiveFieldDeformer::id,
						 &AdaptiveFieldDeformer::creator, &AdaptiveFieldDeformer::initialize,
						 MPxNode::kDeformerNode );
		if (!status) {
			status.perror("registerDeformer");
			return status;
		}
		
        status = plugin.registerCommand("hesperis",
                                        HesperisCmd::creator, HesperisCmd::newSyntax);
        if (!status) {
                status.perror("registerCommand");
                return status;
        }

        return status;
}

MStatus uninitializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj);
		
		status = plugin.deregisterNode( AdaptiveFieldDeformer::id );
		if (!status) {
			  status.perror("deregisterDeformer");
			  return status;
		}

        status = plugin.deregisterCommand("hesperis");
        if (!status) {
                status.perror("deregisterCommand");
                return status;
        }
        
        return status;
}

