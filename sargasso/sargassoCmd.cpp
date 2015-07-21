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
#include <maya/MFnPointArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnVectorArrayData.h>

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
	
	return MS::kSuccess;
}

MStatus SargassoCmd::doIt(const MArgList &argList)
{
	MStatus status;
	status = parseArgs(argList);
	if (!status)
		return status;
	
	return redoIt();
}

MStatus	SargassoCmd::redoIt()
{
	m_sarg = MObject::kNullObj;
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
	
	m_sarg = createNode(transforms, targetMesh);
    
	return MS::kSuccess;
}

MStatus	SargassoCmd::undoIt()
{
	MStatus stat = MGlobal::deleteNode(m_sarg);
	return stat;
}

MStatus SargassoCmd::printHelp()
{
	MGlobal::displayInfo(MString("Sargasso help info:\n binds transforms to a polygonal mesh.")
		+MString("\n howto use sargasso cmd:")
		+MString("\n select a number of transforms, shift-select a mesh")
		+MString("\n run command sargasso"));
	
	return MS::kSuccess;
}

MObject SargassoCmd::createNode(const MObjectArray & transforms,
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
	AHelper::Info<int>(" n transforms: ", nt);
	
	std::map<unsigned, char> bindInds;
	Vector3F * localPs = new Vector3F[nt];
	MVectorArray localPArray;
    MIntArray objTriArray;
	Geometry::ClosestToPointTestResult cls;
	for(i=0;i<nt;i++) {
		MFnTransform ftrans(transforms[i]);
		MVector t = ftrans.getTranslation(MSpace::kTransform);
		MDagPath ptrans;
		MDagPath::getAPathTo(transforms[i], ptrans);
		MMatrix wtm = AHelper::GetWorldTransformMatrix(ptrans);
		t *= wtm;
		
		Vector3F wp(t.x, t.y, t.z);
		//AHelper::Info<unsigned>(" trans ", i);
		//AHelper::Info<Vector3F>(" worldp ", wp);
		
		cls.reset(wp, 1e8f);
		
		tree.closestToPoint(&cls);
		//AHelper::Info<unsigned>(" tri ", cls._icomponent);
		localPs[i] = wp - trimesh.triangleCenter(cls._icomponent);
		localPArray.append(MVector(localPs[i].x, localPs[i].y, localPs[i].z));
		//AHelper::Info<Vector3F>(" localp ", cls._hitPoint);
		bindInds[cls._icomponent] = 1;
        objTriArray.append(cls._icomponent);
	}
	delete[] localPs;
	
	MDGModifier modif;
	MObject osarg = modif.createNode("sargassoNode");
	modif.doIt();
	MFnDependencyNode fsarg(osarg);
	MStatus stat;
	MPlug prestP = fsarg.findPlug("targetRestP", false, &stat);
	
	MFnPointArrayData restPData;
    MObject orestP = restPData.create(ps);
    prestP.setMObject(orestP);
    
	MPlug ptri = fsarg.findPlug("targetTriangle", false, &stat);
	
	MFnIntArrayData triData;
    MObject otri = triData.create(triangleVertices);
    ptri.setMObject(otri);
    
    MPlug plocalP = fsarg.findPlug("objectLocalP", false, &stat);
	
	MFnVectorArrayData localPData;
    MObject olocalP = localPData.create(localPArray);
    plocalP.setMObject(olocalP);
	
	MPlug ptnv = fsarg.findPlug("targetNumV", false, &stat);
	ptnv.setInt(nv);
	
	MPlug ptnt = fsarg.findPlug("targetNumTri", false, &stat);
	ptnt.setInt(triangleVertices.length());
	
	MPlug pobjc = fsarg.findPlug("objectCount", false, &stat);
	pobjc.setInt(nt);
	
	const unsigned nbind = bindInds.size();
	AHelper::Info<unsigned>(" binded to n triangles: ", nbind);
	MIntArray bindTris;
	std::map<unsigned, char>::const_iterator it = bindInds.begin();
	for(;it!=bindInds.end();++it)
		bindTris.append(it->first);
		
	MPlug pbind = fsarg.findPlug("targetBindId", false, &stat);
	
	MFnIntArrayData bindData;
    MObject obind = bindData.create(bindTris);
    pbind.setMObject(obind);
    
    MPlug pobjtri = fsarg.findPlug("objectTriId", false, &stat);
	
	MFnIntArrayData objtriData;
    MObject oobjtri = objtriData.create(objTriArray);
    pobjtri.setMObject(oobjtri);
	
    MPlug pwm = fmesh.findPlug("worldMesh");
    MPlug ptm = fsarg.findPlug("targetMesh");
    MGlobal::executeCommand(MString("connectAttr -f ") +
                            pwm.name() +
                            MString("[0] ") +
                            ptm.name());

    for(i=0;i<nt;i++) {
        MFnTransform ftrans(transforms[i]);
        MGlobal::executeCommand(MString("connectAttr -f ") +
                            ftrans.name() +
                            MString(".parentInverseMatrix[0] ") +
                            fsarg.name() +
                            MString(".constraintParentInvMat[") +
                            i +
                            MString("]"));
        MString ov = fsarg.name() +
                            MString(".outValue[") +
                            i +
                            MString("]");
        MGlobal::executeCommand(MString("connectAttr -f ") +
                            ov +
                            MString(".ctx ") +
                            ftrans.name() +
                            MString(".tx "));
        MGlobal::executeCommand(MString("connectAttr -f ") +
                            ov +
                            MString(".cty ") +
                            ftrans.name() +
                            MString(".ty "));
        MGlobal::executeCommand(MString("connectAttr -f ") +
                            ov +
                            MString(".ctz ") +
                            ftrans.name() +
                            MString(".tz "));
        MGlobal::executeCommand(MString("connectAttr -f ") +
                            ov +
                            MString(".crx ") +
                            ftrans.name() +
                            MString(".rx "));
        MGlobal::executeCommand(MString("connectAttr -f ") +
                            ov +
                            MString(".cry ") +
                            ftrans.name() +
                            MString(".ry "));
        MGlobal::executeCommand(MString("connectAttr -f ") +
                            ov +
                            MString(".crz ") +
                            ftrans.name() +
                            MString(".rz "));
    }
	setResult(fsarg.name());
	return osarg;
}

bool SargassoCmd::isUndoable() const
{ return true; }
