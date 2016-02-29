/*
 *  HesperisCurveIO.cpp
 *  hesperis
 *
 *  Created by jian zhang on 7/12/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HesperisCurveIO.h"
#include <maya/MGlobal.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MPointArray.h>
#include <HWorld.h>
#include <HCurveGroup.h>
#include <CurveGroup.h>
#include <ASearchHelper.h>
#include <boost/format.hpp>
namespace aphid {
    
MObject HesperisCurveCreator::create(CurveGroup * data, MObject & parentObj,
                       const std::string & nodeName)
{
	MObject res = MObject::kNullObj;
	
	if(CheckExistingCurves(data, parentObj)) return res;
	
	MGlobal::displayInfo(MString("hesperis create ")
	+data->numCurves()
    +MString(" curves"));
	
	Vector3F * pnts = data->points();
    unsigned * cnts = data->counts();
    unsigned i=0;
    for(;i<data->numCurves();i++) {
        CreateACurve(pnts, cnts[i], parentObj);
        pnts+= cnts[i];
    }
	
	return res;
}

bool HesperisCurveCreator::CheckExistingCurves(CurveGroup * geos, MObject &target)
{
	MDagPath root;
    MDagPath::getAPathTo(target, root);
	
	MDagPathArray existing;
	ASearchHelper::LsAllTypedPaths(existing, root, MFn::kNurbsCurve);
	
    const unsigned ne = existing.length();
    if(ne < 1) return false;
    if(ne != geos->numCurves()) return false;
    
    unsigned n = 0;
    unsigned i;
    for(i=0; i< ne; i++) {
        MFnNurbsCurve fcurve(existing[i].node());
		n += fcurve.numCVs();
    }
	
    if(n!=geos->numPoints()) {
		AHelper::Info<MString>("existing curves nv don't match cached data ", root.fullPathName());
		return false;
	}
    
    MGlobal::displayInfo(" existing curves matched");
    
    return true;
}

bool HesperisCurveCreator::CreateACurve(Vector3F * pos, unsigned nv, MObject &target)
{
	MPointArray vertexArray;
    unsigned i=0;
	for(; i<nv; i++)
		vertexArray.append( MPoint( pos[i].x, pos[i].y, pos[i].z ) );
	const int degree = 2;
    const int spans = nv - degree;
	const int nknots = spans + 2 * degree - 1;
    MDoubleArray knotSequences;
	knotSequences.append(0.0);
	for(i = 0; i < nknots-2; i++)
		knotSequences.append( (double)i );
	knotSequences.append(nknots-3);
    
    MFnNurbsCurve curveFn;
	MStatus stat;
	curveFn.create(vertexArray,
					knotSequences, degree, 
					MFnNurbsCurve::kOpen, 
					false, false, 
					target, 
					&stat );
					
	return stat == MS::kSuccess;
}

bool HesperisCurveIO::IsCurveValid(const MDagPath & path)
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

bool HesperisCurveIO::CreateCurveGroup(const MDagPathArray & paths, CurveGroup * dst)
{
    MStatus stat;
	unsigned i, j;
	int numCvs = 0;
	unsigned numNodes = 0;
    
	const unsigned n = paths.length();
    for(i=0; i<n; i++) {
		if(!IsCurveValid(paths[i])) continue;
		MFnNurbsCurve fcurve(paths[i].node());
		numCvs += fcurve.numCVs();
		numNodes++;
	}
    
    if(numCvs < 4) {
		MGlobal::displayInfo(" too fews cvs!");
		return false;
	}
    
    dst->create(numNodes, numCvs);
    Vector3F * pnts = dst->points();
	unsigned * counts = dst->counts();
    
    unsigned inode = 0;
	unsigned icv = 0;
	unsigned nj;
	MPoint wp;
	MMatrix worldTm;
	
	for(i=0; i<n; i++) {
		if(!IsCurveValid(paths[i])) continue;
		
		worldTm = GetWorldTransform(paths[i]);
		
		MFnNurbsCurve fcurve(paths[i].node());
		nj = fcurve.numCVs();
		MPointArray ps;
		fcurve.getCVs(ps, MSpace::kWorld);
		
		counts[inode] = nj;
		inode++;
		
		for(j=0; j<nj; j++) {
			wp = ps[j] * worldTm;
			pnts[icv].set((float)wp.x, (float)wp.y, (float)wp.z);
			icv++;
		}
	}
    return true;
}

bool HesperisCurveIO::WriteCurves(const MDagPathArray & paths, 
							HesperisFile * file, 
							const std::string & parentName) 
{
    CurveGroup * gcurve = new CurveGroup;
    if(!CreateCurveGroup(paths, gcurve)) {
        MGlobal::displayInfo(" hesperis check curves error");
        return false;
    }
    
	std::string curveName = "|curves";
    if(parentName.size()>1) curveName = boost::str(boost::format("%1%|curves") % parentName);
	H5PathName(curveName);
    
	MGlobal::displayInfo(MString("hes io write curve group ")+curveName.c_str());
    file->clearCurves();
    file->addCurve(curveName, gcurve);
	
	file->setDirty();
	file->setWriteComponent(HesperisFile::WCurve);
	bool fstat = file->save();
	if(!fstat) MGlobal::displayWarning(MString(" cannot save curves to file ")+ file->fileName().c_str());
	file->close();
	
	return true;
}

bool HesperisCurveIO::ReadCurves(MObject &target)
{
	MGlobal::displayInfo("opium read curve");
    HWorld grpWorld;
    ReadTransformAnd<HCurveGroup, CurveGroup, HesperisCurveCreator>(&grpWorld, target);
    grpWorld.close();
    return true;
}

}
//:~