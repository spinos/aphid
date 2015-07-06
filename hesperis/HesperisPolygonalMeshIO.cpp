#include "HesperisPolygonalMeshIO.h"
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
bool HesperisPolygonalMeshIO::WritePolygonalMeshes(MDagPathArray & paths, HesperisFile * file)
{
    unsigned i = 0;
    for(;i<paths.length();i++)
        MGlobal::displayInfo(MString("todo poly mesh write ")+paths[i].fullPathName());
    return true;
}
