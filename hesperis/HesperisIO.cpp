/*
 *  HesperisIO.cpp
 *  hesperis
 *
 *  Created by jian zhang on 4/21/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisIO.h"
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MFnMesh.h>
#include <maya/MPointArray.h>
#include <maya/MIntArray.h>
#include <CurveGroup.h>
#include <ATriangleMesh.h>
#include "HesperisFile.h"

bool HesperisIO::IsCurveValid(const MDagPath & path)
{
	MStatus stat;
	MFnNurbsCurve fcurve(path.node(), &stat);
	if(!stat) {
		// MGlobal::displayInfo(path.fullPathName() + " is not a curve!");
		return false;
	}
	if(fcurve.numCVs() < 4) {
		MGlobal::displayInfo(path.fullPathName() + " has less than 4 cvs!");
		return false;
	}
	return true;
}

bool HesperisIO::WriteCurves(MDagPathArray & paths, HesperisFile * file) 
{
	MStatus stat;
	const unsigned n = paths.length();
	unsigned i, j;
	int numCvs = 0;
	unsigned numNodes = 0;
	
	MGlobal::displayInfo(" hesperis check curves");
	
	for(i=0; i< n; i++) {
		if(!IsCurveValid(paths[i])) continue;
		MFnNurbsCurve fcurve(paths[i].node());
		numCvs += fcurve.numCVs();
		numNodes++;
	}
	
	if(numCvs < 4) {
		MGlobal::displayInfo(" too fews cvs!");
		return false;
	}
	
	MGlobal::displayInfo(MString(" curve count: ") + numNodes);
	MGlobal::displayInfo(MString(" cv count: ") + numCvs);
	
	CurveGroup gcurve;
	gcurve.create(numNodes, numCvs);
	
	Vector3F * pnts = gcurve.points();
	unsigned * counts = gcurve.counts();
	
	unsigned inode = 0;
	unsigned icv = 0;
	unsigned nj;
	for(i=0; i< n; i++) {
		if(!IsCurveValid(paths[i])) continue;
		
		MFnNurbsCurve fcurve(paths[i].node());
		nj = fcurve.numCVs();
		MPointArray ps;
		fcurve.getCVs(ps, MSpace::kWorld);
		
		counts[inode] = nj;
		inode++;
		
		for(j=0; j<nj; j++) {
			pnts[icv].set((float)ps[j].x, (float)ps[j].y, (float)ps[j].z);
			icv++;
		}
	}
	
	file->setWriteComponent(HesperisFile::WCurve);
	file->addCurve("curves", &gcurve);
	file->setDirty();
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save curves to file ")+ file->fileName().c_str());
	file->close();
	
	return true;
}

bool HesperisIO::WriteMeshes(MDagPathArray & paths, HesperisFile * file)
{
	MStatus stat;
	const unsigned n = paths.length();
	unsigned i, j;
	int numPnts = 0;
	unsigned numNodes = 0;
	unsigned numTris = 0;
	
	MGlobal::displayInfo(" hesperis check meshes");
	
	MIntArray triangleCounts, triangleVertices;
	MPointArray ps;
	
	std::vector<ATriangleMesh * > meshes;
	for(i=0; i< n; i++) {
		MFnMesh fmesh(paths[i].node(), &stat);
		if(!stat) continue;
		numPnts = fmesh.numVertices();
		numNodes++;
		
		fmesh.getTriangles(triangleCounts, triangleVertices);
		numTris = triangleVertices.length() / 3;
		
		MGlobal::displayInfo(paths[i].fullPathName());
		MGlobal::displayInfo(MString(" vertex count: ") + numPnts);
		MGlobal::displayInfo(MString(" triangle count: ") + numTris);
	
		ATriangleMesh * amesh = new ATriangleMesh;
		meshes.push_back(amesh);
		amesh->create(numPnts, numTris);
		
		Vector3F * pnts = amesh->points();
		unsigned * inds = amesh->indices();
	
		fmesh.getPoints(ps, MSpace::kWorld);
			
		for(j=0; j<numPnts; j++)
			pnts[j].set((float)ps[j].x, (float)ps[j].y, (float)ps[j].z);
		
		for(j=0; j<triangleVertices.length(); j++)
			inds[j] = triangleVertices[j];
			
		amesh->setDagName(std::string(paths[i].fullPathName().asChar()));
		std::stringstream sst;
		sst.str("");
		sst<<"mesh"<<numNodes;
		std::string meshName = sst.str();
		file->addTriangleMesh(meshName, amesh);
	}
	
	file->setDirty();
	file->setWriteComponent(HesperisFile::WTri);
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save mesh to file ")+ file->fileName().c_str());
	file->close();
	
	std::vector<ATriangleMesh * >::iterator it = meshes.begin();
	for(;it!=meshes.end();++it) delete *it;
	meshes.clear();
	return true;
}
