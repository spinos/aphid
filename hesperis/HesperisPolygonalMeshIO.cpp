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
bool HesperisPolygonalMeshIO::WritePolygonalMeshes(const std::map<std::string, MDagPath > & paths, HesperisFile * file)
{
    std::vector<APolygonalMesh *> data;
    
    std::map<std::string, MDagPath >::const_iterator it = paths.begin();
    for(;it!=paths.end();++it) {
        APolygonalMesh * mesh = new APolygonalMesh;
        CreateMeshData(mesh, it->second);
        // MGlobal::displayInfo(mesh->verbosestr().c_str());
        file->addPolygonalMesh(it->second.fullPathName().asChar(), mesh);
        data.push_back(mesh);
    }
    
    file->setDirty();
	file->setWriteComponent(HesperisFile::WPoly);
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save poly mesh to file ")+ file->fileName().c_str());
	file->close();
    
    std::vector<APolygonalMesh *>::iterator itd = data.begin();
    for(;itd!=data.end(); ++itd)
        delete *itd;
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
	const int numVertices = data->numPoints();
    
    MObject otm = MObject::kNullObj;
    if(HesperisIO::FindNamedChild(otm, nodeName, parentObj)) {
        if(!checkMeshNv(otm, numVertices))
			MGlobal::displayWarning(MString(" existing node ")+ nodeName.c_str() + 
				MString(" is not a mesh or it is a mesh has wrong number of cvs"));
		
		return otm;
    }
    
    MPointArray vertexArray;
	MIntArray polygonCounts, polygonConnects;
	const int numPolygons = data->numPolygons();
    
    Vector3F * cvs = data->points();
    int i = 0;
    for(;i<numVertices;i++)
        vertexArray.append( MPoint( cvs[i].x, cvs[i].y, cvs[i].z ) );
    
    unsigned * counts = data->faceCounts();
    for(i=0;i<numPolygons;i++)
        polygonCounts.append(counts[i]);
    
    const int numFaceVertices = data->numIndices();
    unsigned * conns = data->indices();
    for(i=0;i<numFaceVertices;i++)
        polygonConnects.append(conns[i]);
    
    MStatus stat;
    MFnMesh fmesh;
	otm = fmesh.create(numVertices, numPolygons, vertexArray, polygonCounts, polygonConnects, parentObj, &stat );

	if(!stat) {
		MGlobal::displayWarning(MString(" hesperis failed to create poly mesh ")+nodeName.c_str());
		return otm;
	}
	
	std::string validName(nodeName);
	SHelper::noColon(validName);
	fmesh.setName(validName.c_str()); 
	
	if(data->numUVs() < 1) {
		MGlobal::displayWarning(MString(" poly mesh has no uv ")+nodeName.c_str());
		return otm;
	}
	
	for(i=0;i<data->numUVs();i++) {
		std::string setName = data->uvName(i);
		addUV(data->uvData(setName), fmesh, setName, polygonCounts);
	}
		
    return otm;
}

void HesperisPolygonalMeshCreator::addUV(APolygonalUV * data, MFnMesh & fmesh,
						const std::string & setName,
						const MIntArray & uvCounts)
{
	MFloatArray uArray, vArray;
	MIntArray uvIds;
	const unsigned ncoord = data->numCoords();
	const unsigned nind = data->numIndices();
	
	float * u = data->ucoord();
	float * v = data->vcoord(); 
	unsigned i = 0;
	for(;i<ncoord;i++) {
		uArray.append(u[i]);
		vArray.append(v[i]);
	}
	
	unsigned * ind = data->indices();
	for(i=0; i<nind; i++)
		uvIds.append(ind[i]);
		
	const MString uvSet(setName.c_str());
	MStatus stat = fmesh.setUVs( uArray, vArray, &uvSet);
	if(!stat) {
		MGlobal::displayWarning(MString(" hesperis failed to create uv set coord ")+uvSet);
		return;
	}
	
	stat = fmesh.assignUVs( uvCounts, uvIds, &uvSet );
	if(!stat)
		MGlobal::displayWarning(MString(" hesperis failed to create uv set uvid ")+uvSet);
		
	return;
}

bool HesperisPolygonalMeshCreator::checkMeshNv(const MObject & node, unsigned nv)
{
	MStatus stat;
	MFnMesh fmesh(node, &stat);
	if(!stat)
		return false;
	
	return (fmesh.numVertices() == nv);
}
//:~