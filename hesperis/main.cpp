#include <maya/MFnPlugin.h>
#include <maya/MGlobal.h>
#include "HesperisCmd.h"

MStatus initializePlugin(MObject obj)
{
        MStatus         status;
        MFnPlugin       plugin(obj, "Zhang Jian", "1.2 Fri Jun 5 18:33:27 2015", "Any");

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

