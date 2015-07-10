#include "HesperisPolygonalMeshIO.h"
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <maya/MItMeshPolygon.h>
#include <APolygonalMesh.h>
#include <APolygonalUV.h>
#include "HesperisFile.h"
#include <HWorld.h>
#include <SHelper.h>
#include <HTransform.h>
#include <HPolygonalMesh.h>
bool HesperisPolygonalMeshIO::WritePolygonalMeshes(MDagPathArray & paths, HesperisFile * file)
{
    std::vector<APolygonalMesh *> data;
    
    unsigned i = 0;
    for(;i<paths.length();i++) {
        APolygonalMesh * mesh = new APolygonalMesh;
        CreateMeshData(mesh, paths[i]);
        // MGlobal::displayInfo(mesh->verbosestr().c_str());
        file->addPolygonalMesh(paths[i].fullPathName().asChar(), mesh);
        data.push_back(mesh);
    }
    
    file->setDirty();
	file->setWriteComponent(HesperisFile::WPoly);
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save poly mesh to file ")+ file->fileName().c_str());
	file->close();
    
    std::vector<APolygonalMesh *>::iterator it = data.begin();
    for(;it!=data.end(); ++it)
        delete *it;
    data.clear();
    return true;
}

bool HesperisPolygonalMeshIO::CreateMeshData(APolygonalMesh * data, const MDagPath & path)
{
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
	
	if(fmesh.numUVSets() < 1) {
		MGlobal::displayWarning(MString(" mesh has no uv ")+path.fullPathName());
		return true;
	}
	
	MStringArray setNames;
	fmesh.getUVSetNames(setNames);
	
	for(i=0; i< setNames.length(); i++) {
		APolygonalUV * auv = new APolygonalUV;
		CreateMeshUV(auv, path, setNames[i]);
		data->addUV(setNames[i].asChar(), auv);
	}
	
    return true;
}

bool HesperisPolygonalMeshIO::CreateMeshUV(APolygonalUV * data, const MDagPath & path, const MString & setName)
{
	MFloatArray uarray, varray;
    MIntArray uvIds;
	
	MFnMesh fmesh(path.node());
	fmesh.getUVs( uarray, varray, &setName );
	
	MItMeshPolygon faceIt(path);
	for( ; !faceIt.isDone(); faceIt.next() ) {
        for( int k=0; k < faceIt.polygonVertexCount(); k++ ) {
            int aid;
            faceIt.getUVIndex( k, aid, &setName );
            uvIds.append(aid);
        }
    }
	
	unsigned ncoords = uarray.length();
	unsigned ninds = uvIds.length();
	
	data->create(ncoords, ninds);
	
	float * u = data->ucoord();
	float * v = data->vcoord();
	unsigned * ind = data->indices();
	
	unsigned i;
	for(i=0; i< ncoords; i++) {
		u[i] = uarray[i];
		v[i] = varray[i];
	}
	
	for(i=0; i< ninds; i++) 
		ind[i] = uvIds[i];

	return true;
}

bool HesperisPolygonalMeshIO::ReadMeshes(HesperisFile * file, MObject &target)
{
    MGlobal::displayInfo("opium read poly mesh");
    HWorld grpWorld;
    ReadTransformAnd<HPolygonalMesh, APolygonalMesh, HesperisPolygonalMeshCreator>(&grpWorld, target);
    grpWorld.close();
    return true;
}

MObject HesperisPolygonalMeshCreator::create(APolygonalMesh * data, MObject & parentObj,
                       const std::string & nodeName)
{
    MGlobal::displayInfo(MString("todo poly mesh in ")+nodeName.c_str());         
    MGlobal::displayInfo(data->verbosestr().c_str());

    MObject otm = MObject::kNullObj;
    if(!HesperisIO::FindNamedChild(otm, nodeName, parentObj)) {
        MGlobal::displayInfo(MString(" node doesn't exist ")+nodeName.c_str());
        
    } 
    return otm;
}
//:~