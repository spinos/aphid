/*
 *  FrustumBoxCmd.cpp
 *  FrustumBox
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 *  frustumBox -fr 1 24 -lv -cms 1.3 -cam |persp'perspShape -cms 1.1
 *  default cameraScale 1.0
 */

#include "FrustumBoxCmd.h"
#include <mama/ASearchHelper.h>
#include <mama/AHelper.h>
#include <mama/MeshHelper.h>
#include <maya/MFnPointArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnAnimCurve.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MFnCamera.h>
#include <maya/MItMeshPolygon.h>
#include <GjkIntersection.h>
#include <geom/PrincipalComponents.h>

#define kCameraScaleFlag "-cms" 
#define kCameraScaleFlagLong "-cameraScale"
#define kTimeStepFlag "-ts" 
#define kTimeStepFlagLong "-timeStep"

using namespace aphid;

FrustumBoxCmd::FrustumBoxCmd() {}
FrustumBoxCmd::~FrustumBoxCmd() {}

void* FrustumBoxCmd::creator()
{
	return new FrustumBoxCmd;
}

MSyntax FrustumBoxCmd::newSyntax() 
{
	MSyntax syntax;

	syntax.addFlag("-h", "-help", MSyntax::kNoArg);
    syntax.addFlag("-fr", "-frameRange", MSyntax::kLong, MSyntax::kLong);
    syntax.addFlag("-ts", "-timeStep", MSyntax::kLong);
	syntax.addFlag("-cam", "-camera", MSyntax::kString);
    syntax.addFlag("-lv", "-listVisible", MSyntax::kNoArg);
	syntax.addFlag(kCameraScaleFlag, kCameraScaleFlagLong, MSyntax::kDouble);
	syntax.enableQuery(false);
	syntax.enableEdit(false);

	return syntax;
}

MStatus FrustumBoxCmd::parseArgs(const MArgList &args)
{
	m_cameraScale = 1.0;
	m_timeStep = 1;
	
	MStatus			stat;
	MArgDatabase	argData(syntax(), args, &stat);

	if ( !stat )
		return MS::kFailure;

	m_mode = WHelp;
	m_cameraName = "";
    m_startTime = 1;
    m_endTime = 24;
    m_doListVisible = false;
	
    if(argData.isFlagSet("-fr")) {
        m_mode = WAction;
        stat = argData.getFlagArgument("-fr", 0, m_startTime);
        stat = argData.getFlagArgument("-fr", 1, m_endTime);
        if(!stat) {
            MGlobal::displayWarning("invalid -fr argument");
            m_mode = WHelp;
        }
    }
    
    if(argData.isFlagSet(kTimeStepFlag)) {
        stat = argData.getFlagArgument(kTimeStepFlag, 0, m_timeStep);
        if(!stat) {
            MGlobal::displayWarning("invalid -ts argument");
            m_mode = WHelp;
        }
        
        if(m_timeStep<1) {
            m_timeStep = 1;
            AHelper::Info<int>("FrustumBoxCmd parseArgs truncate timeStep to ", m_timeStep);
        }
    }
	
	if(argData.isFlagSet("-cam")) {
        m_mode = WAction;
        stat = argData.getFlagArgument("-cam", 0, m_cameraName);
        if(!stat) {
            MGlobal::displayWarning("invalid -cam argument");
            m_mode = WHelp;
        }
    }
    
    if(argData.isFlagSet("-lv")) m_doListVisible = true;
	
	if(argData.isFlagSet(kCameraScaleFlag)) {
		stat = argData.getFlagArgument(kCameraScaleFlag, 0, m_cameraScale);
		if (!stat) {
			MGlobal::displayWarning(" frustumBox cannot parse -cms flag");
		}
		
		if(m_cameraScale < 0.1) {
			MGlobal::displayWarning(" frustumBox -cms truncated to 0.1");
			m_cameraScale = 0.1;
		}
	}
    
	if(argData.isFlagSet("-h")) m_mode = WHelp;
	
	if(m_cameraName.length() < 3) {
		MGlobal::displayWarning("invalid camera name");
		m_mode = WHelp;
	}
	
	return MS::kSuccess;
}

MStatus FrustumBoxCmd::doIt(const MArgList &argList)
{
	MStatus status;
	status = parseArgs(argList);
	if (!status)
		return status;
	
	return redoIt();
}

MStatus	FrustumBoxCmd::redoIt()
{
	if(m_mode == WHelp) return printHelp();
    
    MObject ocamera;
    ASearchHelper finder;
    if( !finder.getObjByFullName(m_cameraName.asChar(), ocamera) ) {
        MGlobal::displayWarning(MString("cannot find camera ")
                                +m_cameraName);
        return MS::kSuccess;
    }
    
    MStatus stat;
    MFnCamera fcam(ocamera, &stat);
    if(!stat) {
        MGlobal::displayWarning(MString("not a camera ")
                                +m_cameraName);
        return stat;
    }
    
	MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if(selList.length() < 1) {
		MGlobal::displayWarning("empty selection");
		return MS::kSuccess;
	}
    
    MGlobal::displayInfo(MString("test camera frustum box intersection from ")
                         + m_startTime
                         + " to "
                         + m_endTime);
	
    MDagPathArray paths;
    MItSelectionList iter( selList );
	for ( ; !iter.isDone(); iter.next() ) {								
		MDagPath apath;		
		iter.getDagPath( apath );
        paths.append(apath);
	}
    
    AHelper::Info<unsigned>("n objects ", paths.length());
    std::vector<int> visibilities;
    int i;
    for(i=0; i< paths.length(); i++) visibilities.push_back(0);
    
    for(i=m_startTime; i<= m_endTime; i += m_timeStep) {
        MGlobal::executeCommand(MString("currentTime ") + i);
        collide(visibilities,
                 paths,
                 ocamera);
        MGlobal::displayInfo(MString("frame") + i);
        if(isAllVisible(visibilities)) {
            MGlobal::displayInfo("all objects are visible");
            break;
        }
    }

    MGlobal::displayInfo(MString("done "));
    
    MStringArray names;
    listObjects(names, paths, visibilities, m_doListVisible);
    setResult(names);
    
	return MS::kSuccess;
}

MStatus FrustumBoxCmd::printHelp()
{
	MGlobal::displayInfo(MString("FrustumBox help info:\n test if objects are visible through a camera")
		+MString("\n by intersecting object oriented box and camera frustum. \n example:")
		+MString("\n select object(s) to test")
		+MString("\n frustumBox -cam |persp|perspShape -fr 1 100 -ts 5")
		+MString("\n -cam/-camera string is full path to camera shape")
		+MString("\n -fr/-frameRange int int is begin and end frame of test")
        +MString("\n -ts/-timeStep int is number of frames of each step within frame range, default is 1")
        +MString("\n -lv/-listVisible is set to return visible objects instead of invisible ones")
		+MString("\n -cms/-cameraScale double scaling focal length of camera to expand or shrink region of contact default 1")
		+MString("\n return value is string[] of objects NOT visible to camera"));
	
	return MS::kSuccess;
}

void FrustumBoxCmd::collide(std::vector<int> & visibilities,
                 const MDagPathArray & paths,
                 MObject & camera) const
{
    const unsigned n = paths.length();
    Vector3F frustumA[8];
    worldFrustum(frustumA, camera);
    
    gjk::IntersectTest test;
    test.SetABox(frustumA);

    gjk::BoxSet boxB;
    unsigned i;
    for(i=0; i< n; i++) {
		worldBBox(boxB.x(), paths[i]);
		if(test.evaluate(&boxB)) {
/// test oriented box
			worldOrientedBox(boxB.x(), paths[i]);
			if(test.evaluate(&boxB)) {
				// MGlobal::displayInfo("intersected"); 
				visibilities[i] = 1;
			}
		}
    }
}

void FrustumBoxCmd::worldOrientedBox(aphid::Vector3F * corners, const MDagPath & path) const
{
    AHelper::Info<const char *>("get world oriented box", path.fullPathName().asChar() );
	MDagPathArray meshPaths;
	ASearchHelper::LsAllTypedPaths(meshPaths, path, MFn::kMesh);

	const int n = meshPaths.length();
	if(n < 1) {
		AHelper::Info<MString>(" FrustumBoxCmd find no mesh in path, use world bbox", path.fullPathName() );
		worldBBox(corners, path);
		return;
	}
	
	sdb::VectorArray<cvx::Triangle> tris;
	BoundingBox bbox;
	
	int i=0;
	for(;i<n;++i) {
		MeshHelper::GetMeshTriangles(tris, bbox, meshPaths[i], path);
	}
	
	const int nt = tris.size();
	if(nt < 1) {
		AHelper::Info<MString>(" FrustumBoxCmd find no mesh triangle in path, use world bbox", path.fullPathName() );
		worldBBox(corners, path);
		return;
	}
	
	std::vector<Vector3F> pnts;
    for(i=0; i< nt; ++i) {
        const aphid::cvx::Triangle * t = tris[i];
/// at triangle center
        pnts.push_back(t->P(0) * .33f 
                       + t->P(1) * .33f
                       + t->P(2) * .33f);
    }
	
	tris.clear();
	
	PrincipalComponents<std::vector<Vector3F> > obpca;
    AOrientedBox obox = obpca.analyze(pnts, pnts.size() );
	
	pnts.clear();
	
	obox.getBoxVertices(corners);
	
}

void FrustumBoxCmd::worldBBox(Vector3F * corners, const MDagPath & path) const
{
    MBoundingBox res;
    MStatus stat;
    MFnDagNode fnode(path, &stat);
    res = fnode.boundingBox();
    
    const MPoint pmin = res.min();
    const MPoint pmax = res.max();
    corners[0].set(pmin.x, pmin.y, pmin.z);
    corners[1].set(pmax.x, pmin.y, pmin.z);
    corners[2].set(pmin.x, pmax.y, pmin.z);
    corners[3].set(pmax.x, pmax.y, pmin.z);
    corners[4].set(pmin.x, pmin.y, pmax.z);
    corners[5].set(pmax.x, pmin.y, pmax.z);
    corners[6].set(pmin.x, pmax.y, pmax.z);
    corners[7].set(pmax.x, pmax.y, pmax.z);
    
    MMatrix m = AHelper::GetWorldParentTransformMatrix(path);
    Matrix44F mat;
    AHelper::ConvertToMatrix44F(mat, m);
    int i;
    for(i=0; i<8; i++) corners[i] = mat.transform(corners[i]);
    // for(i=0; i<8; i++) AHelper::Info<Vector3F>("boxB", corners[i]);
}

void FrustumBoxCmd::worldFrustum(Vector3F * corners, MObject & camera) const
{
    MStatus stat;
    MFnCamera fcam(camera, &stat);
    
    const double nearClip = fcam.nearClippingPlane() + 0.001f;
    const double farClip = fcam.farClippingPlane() - 0.001f;
    const double hAperture = fcam.horizontalFilmAperture();
    const double vAperture = fcam.verticalFilmAperture();
	double fl = fcam.focalLength();
	if(m_cameraScale != 1.0) {
		fl /= m_cameraScale;
		AHelper::Info<double>("scale focal length by", 1.0/m_cameraScale);
	}
    
    float h_fov = hAperture * 0.5 / ( fl * 0.03937 );
    float v_fov = vAperture * 0.5 / ( fl * 0.03937 );

    float fright = farClip * h_fov;
    float ftop = farClip * v_fov;

    float nright = nearClip * h_fov;
    float ntop = nearClip * v_fov;

    corners[0].set(fright, ftop, -farClip);
	corners[1].set(-fright, ftop, -farClip);
	corners[2].set(-fright, -ftop, -farClip);
	corners[3].set(fright, -ftop, -farClip);
	corners[4].set(nright, ntop, -nearClip);
	corners[5].set(-nright, ntop, -nearClip);
	corners[6].set(-nright, -ntop, -nearClip);
	corners[7].set(nright, -ntop, -nearClip);
    
    MDagPath pcam;
    MDagPath::getAPathTo(camera, pcam);
    
    MMatrix m = AHelper::GetWorldTransformMatrix(pcam);
    Matrix44F mat;
    AHelper::ConvertToMatrix44F(mat, m);
    
    // AHelper::Info<Matrix44F>("cammat", mat);
    
    int i;
    for(i=0; i<8; i++) corners[i] = mat.transform(corners[i]);
    // for(i=0; i<8; i++) AHelper::Info<Vector3F>("frustumA", corners[i]);
}

bool FrustumBoxCmd::isAllVisible(const std::vector<int> & visibilities) const
{
    unsigned i;
    const unsigned n = visibilities.size();
    for(i=0; i< n; i++) {
        if(visibilities[i] == 0) return false;   
    }
    return true;
}

void FrustumBoxCmd::listObjects(MStringArray & dst,
                 const MDagPathArray & paths,
                 const std::vector<int> & visibilities,
                 bool visible)
{
    int k = 0;
    if(visible) k = 1;
    unsigned i;
    const unsigned n = visibilities.size();
    for(i=0; i< n; i++) {
        if(visibilities[i] == k) {
            dst.append(paths[i].fullPathName());
        }
    }
}
//:~