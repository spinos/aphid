#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "keychangeCallback.h"

MStatus initializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj, "Zhang Jian", "0.0 Fri Apr 10 19:55:33 CST 2015", "Any");

        status = plugin.registerCommand("keychangeCallbacks",
                                        keychagneCallbacks::creator);
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

        status = plugin.deregisterCommand("keychangeCallbacks");
        if (!status) {
                status.perror("deregisterCommand");
                return status;
        }
        
        // Remove callbacks when plug-in is unloaded.
        //MMessage::removeCallback(addIK2BsolverCallbacks::afterNewId);
        //MMessage::removeCallback(addIK2BsolverCallbacks::afterOpenId);

        return status;
}

