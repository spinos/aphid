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
#include <maya/MPointArray.h>
#include <CurveGroup.h>
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

bool HesperisIO::WriteCurves(MDagPathArray & paths, const std::string & fileName) 
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
	
	HesperisFile hesf;
	bool fstat = hesf.create(fileName.c_str());
	if(!fstat) MGlobal::displayWarning(MString(" cannot create file ")+ fileName.c_str());
	hesf.addCurve("curves", &gcurve);
	hesf.setDirty();
	fstat = hesf.save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save file ")+ fileName.c_str());
	hesf.close();
	MGlobal::displayInfo(" done.");
	return true;
}
