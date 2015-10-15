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
bool HesperisCurveIO::ReadCurves(HesperisFile * file, MObject &target)
{
	MGlobal::displayInfo("opium read curve");
    HWorld grpWorld;
    ReadTransformAnd<HCurveGroup, CurveGroup, HesperisCurveCreator>(&grpWorld, target);
    grpWorld.close();
    return true;
}

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
