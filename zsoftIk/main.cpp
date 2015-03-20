#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "zsoftIkSolver.h"
/////////////////////////////////////////////////////////
//
// Register the IK 2 Bone Solver and Callback Command
//
/////////////////////////////////////////////////////////
MStatus initializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj, "Autodesk", "2.5", "Any");

        status = plugin.registerNode("zik2Bsolver", 
                                                                 ik2Bsolver::id,
                                                                 &ik2Bsolver::creator,
                                                                 &ik2Bsolver::initialize,
                                                                 MPxNode::kIkSolverNode);
        if (!status) {
                status.perror("registerNode");
                return status;
        }

        status = plugin.registerCommand("addZIK2BsolverCallbacks",
                                        addIK2BsolverCallbacks::creator);
        if (!status) {
                status.perror("registerCommand");
                return status;
        }

        // Register post-load MEL proc
        //
        // Note: We make use of the MFnPlugin::registerUI() method which executes
        // the given MEL procedures following the plugin load to execute our method.
        // This method will ensure that the solver node is created on plugin load.
        //
        //status = plugin.registerUI("ik2Bsolver", "");

        return status;
}

MStatus uninitializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj);

        status = plugin.deregisterNode(ik2Bsolver::id);
        if (!status) {
                status.perror("deregisterNode");
                return status;
        }

        status = plugin.deregisterCommand("addZIK2BsolverCallbacks");
        if (!status) {
                status.perror("deregisterCommand");
                return status;
        }
        
        // Remove callbacks when plug-in is unloaded.
        MMessage::removeCallback(addIK2BsolverCallbacks::afterNewId);
        MMessage::removeCallback(addIK2BsolverCallbacks::afterOpenId);

        return status;
}

