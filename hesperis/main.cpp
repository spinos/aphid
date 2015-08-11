#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "HesperisCmd.h"
#include "AdaptiveFieldDeformer.h"
static const MString Verinfo("1.3.5 Tue Aug 11 17:33:15 CST 2015 field deform");
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
		
        status = plugin.registerCommand("hesperis",
                                        HesperisCmd::creator, HesperisCmd::newSyntax);
        if (!status) {
                status.perror("registerCommand");
                return status;
        }

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

        status = plugin.deregisterCommand("hesperis");
        if (!status) {
                status.perror("deregisterCommand");
                return status;
        }
        
        MGlobal::displayInfo(MString("hesperis unload plug-in version ") + Verinfo);
        
        return status;
}

