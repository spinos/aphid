/*
 *  rotaCmd.cpp
 *  rotaConstraint
 *
 *  Created by jian zhang on 7/7/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "sargassoCmd.h"
#include "sargassoNode.h"
#include <ASearchHelper.h>
#include <ATriangleMesh.h>
#include <KdTree.h>

SargassoCmd::SargassoCmd() {}
SargassoCmd::~SargassoCmd() {}

void* SargassoCmd::creator()
{
	return new SargassoCmd;
}

MSyntax SargassoCmd::newSyntax() 
{
	MSyntax syntax;

	syntax.addFlag("-h", "-help", MSyntax::kNoArg);
	syntax.enableQuery(false);
	syntax.enableEdit(false);

	return syntax;
}

MStatus SargassoCmd::parseArgs(const MArgList &args)
{
	MStatus			ReturnStatus;
	MArgDatabase	argData(syntax(), args, &ReturnStatus);

	if ( ReturnStatus.error() )
		return MS::kFailure;

	m_mode = WCreate;
	
	if(argData.isFlagSet("-h")) m_mode = WHelp;
	
	return MS::kUnknownParameter;
}

MStatus SargassoCmd::doIt(const MArgList &argList)
{
	MStatus ReturnStatus;

	if ( MS::kFailure == parseArgs(argList) )
		return MS::kFailure;
		
	if(m_mode == WHelp) return printHelp();
	
	MSelectionList selList;
    MGlobal::getActiveSelectionList(selList);
    
	const unsigned nsel = selList.length();
	if(nsel < 2) {
		MGlobal::displayInfo(" Insufficient selction!");
		return printHelp();
	}
	
	MObjectArray transforms;
	MObject targetMesh;
	
	unsigned i = 0;
	MItSelectionList iter( selList );
	for(; !iter.isDone(); iter.next()) {								
		MDagPath apath;		
		iter.getDagPath( apath );
		if(i<nsel-1) {
			if(apath.node().hasFn(MFn::kTransform))
				transforms.append(apath.node());
		}
		else {
			if(apath.node().hasFn(MFn::kMesh))
				targetMesh = apath.node();
			else
				ASearchHelper::FirstTypedObj(apath.node(), targetMesh, MFn::kMesh);
		}
		i++;
	}
	
	if(transforms.length() < 1 || targetMesh.isNull()) {
		MGlobal::displayInfo(" Insufficient selction! ");
		return printHelp();
	}
	
	createNode(transforms, targetMesh);
    
	return MS::kUnknownParameter;
}

MStatus SargassoCmd::printHelp()
{
	MGlobal::displayInfo(MString("Sargasso help info:\n binds transforms to a polygonal mesh.")
		+MString("\n howto use sargasso cmd:")
		+MString("\n select a number of transforms, shift-select a mesh")
		+MString("\n run command sargasso"));
	
	return MS::kSuccess;
}

MStatus SargassoCmd::createNode(const MObjectArray & transforms,
					const MObject & targetMesh)
{
	MFnMesh fmesh(targetMesh);
	AHelper::Info<MString>(" target mesh: ", fmesh.name());
	
	MDagPath pmesh;
	MDagPath::getAPathTo(targetMesh, pmesh);
	MMatrix wm = AHelper::GetWorldTransformMatrix(pmesh);
	
	MPointArray ps;
	fmesh.getPoints(ps);
	
	const unsigned nv = ps.length();
	unsigned i = 0;
	for(;i<nv;i++) ps[i] *= wm;
	
	MIntArray triangleCounts, triangleVertices;
	fmesh.getTriangles(triangleCounts, triangleVertices);
	
	ATriangleMesh trimesh;
	trimesh.create(nv, triangleVertices.length()/3);
	
	Vector3F * cvs = trimesh.points();
	unsigned * ind = trimesh.indices();
	for(i=0;i<nv;i++) cvs[i].set(ps[i].x, ps[i].y, ps[i].z);
	for(i=0;i<triangleVertices.length();i++) ind[i] = triangleVertices[i];
	
	AHelper::Info<std::string>(" target ", trimesh.verbosestr());
	
	KdTree::MaxBuildLevel = 20;
	KdTree::NumPrimitivesInLeafThreashold = 9;
	
	KdTree tree;
	tree.addGeometry(&trimesh);
	tree.create();
	
	const unsigned nt = transforms.length();
	AHelper::Info<int>(" to bind n transforms: ", nt);
	
	Vector3F * localPs = new Vector3F[nt];
	Geometry::ClosestToPointTestResult cls;
	for(i=0;i<nt;i++) {
		MFnTransform ftrans(transforms[i]);
		MVector t = ftrans.getTranslation(MSpace::kTransform);
		MDagPath ptrans;
		MDagPath::getAPathTo(transforms[i], ptrans);
		MMatrix wtm = AHelper::GetWorldTransformMatrix(ptrans);
		t *= wtm;
		
		Vector3F wp(t.x, t.y, t.z);
		AHelper::Info<unsigned>(" trans ", i);
		AHelper::Info<Vector3F>(" worldp ", wp);
		
		cls.reset(wp, 1e8f);
		
		tree.closestToPoint(&cls);
		AHelper::Info<unsigned>(" tri ", cls._icomponent);
		localPs[i] = wp - trimesh.triangleCenter(cls._icomponent);
		AHelper::Info<Vector3F>(" localp ", cls._hitPoint);
	}
	
	delete[] localPs;
	return MS::kSuccess;
}
