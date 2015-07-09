#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "HesperisCmd.h"

MStatus initializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj, "Zhang Jian", "1.3.3 Fri Jul 10 3:28:42 2015 grow mesh in world space", "Any");

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

        status = plugin.deregisterCommand("hesperis");
        if (!status) {
                status.perror("deregisterCommand");
                return status;
        }
        
        return status;
}

