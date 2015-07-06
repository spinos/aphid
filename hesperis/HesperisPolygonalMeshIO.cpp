#include "HesperisPolygonalMeshIO.h"
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MItMeshPolygon.h>
#include <APolygonalMesh.h>
bool HesperisPolygonalMeshIO::WritePolygonalMeshes(MDagPathArray & paths, HesperisFile * file)
{
    std::vector<APolygonalMesh *> data;
    
    unsigned i = 0;
    for(;i<paths.length();i++) {
        APolygonalMesh * mesh = new APolygonalMesh;
        CreateMeshData(mesh, paths[i]);
        data.push_back(mesh);
    }
    
    std::vector<APolygonalMesh *>::iterator it = data.begin();
    for(;it!=data.end(); ++it)
        delete *it;
    data.clear();
    return true;
}

bool HesperisPolygonalMeshIO::CreateMeshData(APolygonalMesh * data, const MDagPath & path)
{
    MGlobal::displayInfo(MString("todo poly mesh write ")+path.fullPathName());
    MStatus stat;
    MFnMesh fmesh(path.node(), &stat);
    if(!stat) {
        MGlobal::displayInfo(MString(" not a mesh ") + path.fullPathName());
        return false;
    }
    
    unsigned np = fmesh.numVertices();
    unsigned nf = fmesh.numPolygons();
    unsigned ni = fmesh.numFaceVertices();
    
    data->create(np, ni, nf);
    Vector3F * pnts = data->points();
	unsigned * inds = data->indices();
    unsigned * cnts = data->faceCounts();
    
    MPointArray ps;
    MPoint wp;
	MMatrix worldTm;
    
    worldTm = GetWorldTransform(path);
    fmesh.getPoints(ps, MSpace::kObject);
	
    unsigned i = 0;
    for(;i<np;i++) {
        wp  = ps[i] * worldTm;
        pnts[i].set((float)wp.x, (float)wp.y, (float)wp.z);
    }
    
    unsigned j;
    unsigned acc = 0;
    MIntArray vertices;
    MItMeshPolygon faceIt(path);
    for(i=0; !faceIt.isDone(); faceIt.next(), i++) {
        cnts[i] = faceIt.polygonVertexCount();
        faceIt.getVertices(vertices);
        for(j = 0; j < vertices.length(); j++) {
            inds[acc] = vertices[j];
            acc++;
        }
    }
    
    data->computeFaceDrift();
    return true;
}
