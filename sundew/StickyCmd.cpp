/*
 * select mesh(s) or mesh vertex(ices)
 * run command: sticky
*/

#include "StickyCmd.h"
#include <maya/MArgDatabase.h>
#include <maya/MArgList.h>
#include <maya/MFnCamera.h>
#include <maya/MItCurveCV.h>
#include <maya/MFnNurbsCurve.h>
#include <maya/MPointArray.h>
#include <maya/MDagModifier.h>
#include <maya/MDGModifier.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MItMeshVertex.h>
#include <ASearchHelper.h>

StickyCmd::StickyCmd()
{
	setCommandString("StickyCmd");
}

StickyCmd::~StickyCmd() {}

void* StickyCmd::creator()
{
	return new StickyCmd;
}

MSyntax StickyCmd::newSyntax()
{
	MSyntax syntax;
	return syntax;
}

MStatus StickyCmd::doIt(const MArgList &args)
{
	MStatus status = parseArgs(args);

	MSelectionList sels;
	MGlobal::getActiveSelectionList ( sels );
	
	MItSelectionList iter( sels );

	float mass = 0.f;
	MPoint mean(0.0,0.0,0.0);
    for ( ; !iter.isDone(); iter.next() ) {								
        MDagPath item;			
        MObject component;		
        iter.getDagPath( item, component );
		addPosition(&mean, &mass, item, component);
    }
	
	if(mass < 1.f) {
		AHelper::Info<float>("empty selection ", mass );
		return status;
	}
	
	AHelper::Info<float>("total mass", mass);
	mean = mean * (1.f/mass);
	AHelper::Info<MPoint>("mean pos", mean);
	
	MDagPath closestMesh;
	unsigned closestVert;
	float minD = 1e8;
	iter.reset();
	for ( ; !iter.isDone(); iter.next() ) {								
        MDagPath item;			
        MObject component;		
        iter.getDagPath( item, component );
		getVertClosestToMean(&minD, closestMesh, closestVert, item, component, mean);
    }
	
	int meshId = 0;
	iter.reset();
	int i=0;
	for ( ; !iter.isDone(); iter.next() ) {								
        MDagPath item;			
        MObject component;		
        iter.getDagPath( item, component );
		if(closestMesh == item) {
			meshId = i;
		}
		i++;
    }
	AHelper::Info<MString>("choose mesh", closestMesh.fullPathName() );
	AHelper::Info<int>("mesh id", meshId);
	AHelper::Info<unsigned>("choose vert", closestVert );
	
	MGlobal::setActiveSelectionList(sels);
	MObject deformer = createDeformer();
	if(deformer.isNull() ) return status;
	
	MObject viz = connectViz(closestVert);
	if(viz.isNull() ) return status;
	
	storeRef(viz, closestMesh, closestVert);
	
	connectDeformer(viz, deformer, meshId);
	return status;
}

MStatus StickyCmd::parseArgs(const MArgList &args)
{
	MStatus status;
	MArgDatabase argData(syntax(), args);
	
	return MStatus::kSuccess;
}

void StickyCmd::addPosition(MPoint * sum, float * mass, const MDagPath & mesh, MObject & vert)
{
	MStatus stat;
	MItMeshVertex itmv(mesh, vert, &stat);
	if(!stat) {
		AHelper::Info<MString>("sticky error not a mesh ", mesh.fullPathName() );
		return;
	}
	
	for(;!itmv.isDone();itmv.next() ) {
		*sum += itmv.position();
		*mass += 1.f;
	}
}

void StickyCmd::getVertClosestToMean(float *minD, MDagPath & closestMesh, unsigned & closestVert, 
									const MDagPath & mesh, MObject & vert, const MPoint & mean)
{
	MStatus stat;
	MItMeshVertex itmv(mesh, vert, &stat);
	if(!stat) {
		AHelper::Info<MString>("sticky error not a mesh ", mesh.fullPathName() );
		return;
	}
	
	for(;!itmv.isDone();itmv.next() ) {
		MVector d = mean - itmv.position();
		float l = d.length();
		if(*minD > l) {
			*minD = l;
			closestMesh = mesh;
			closestVert = itmv.index();
		}
	}
}

MObject StickyCmd::connectViz(unsigned vert)
{	
	MDagModifier mod;
	MObject tm = mod.createNode("transform");
	MStatus stat = mod.doIt();
	MObject viz = mod.createNode("stickyLocator", tm);
	stat = mod.doIt();

	if(viz.isNull() ) {
		AHelper::Info<MString>("sticky error null viz ", "" );
		return MObject::kNullObj;
	}
	
	MFnDependencyNode fviz(viz);
	MPlug vidPlug = fviz.findPlug("vertexId", false, &stat);
	if(!stat) {
		AHelper::Info<MString>("sticky error cannot find vertex id attr ", fviz.name() );
		return MObject::kNullObj;
	}
	vidPlug.setValue((int)vert);
	
	AHelper::Info<MString>("stick create viz", fviz.name() );
	return viz;
}

MObject StickyCmd::createDeformer()
{
	MStringArray rdef;
	MStatus stat = MGlobal::executeCommand( MString("deformer -type stickyDeformer "),  rdef);
	if(!stat) {
		AHelper::Info<MString>("sticky error cannot create deformer ", "stickyDeformer" );
		return MObject::kNullObj;
	}
	
	MGlobal::selectByName (rdef[0], MGlobal::kReplaceList);
	MSelectionList sels;
	MGlobal::getActiveSelectionList ( sels );
	
	MObject node;
	MItSelectionList iter( sels );
	iter.getDependNode(node);
	
	if(node.isNull() ) {
		AHelper::Info<MString>("sticky error null deformer ", "stickyDeformer" );
		return MObject::kNullObj;
	}
	
	MFnDependencyNode fnode(node);
	
	AHelper::Info<MString>("stick create deformer", fnode.name() );
	
	return node;
}

void StickyCmd::connectDeformer(const MObject & viz, const MObject & deformer, int meshId)
{
	MFnDependencyNode fviz(viz);
	MFnDependencyNode fdeformer(deformer);
	MDGModifier mod;
	mod.connect(fviz.findPlug("size"), fdeformer.findPlug("radius"));
	mod.connect(fviz.findPlug("displaceVec"), fdeformer.findPlug("inVec"));
	mod.connect(fviz.findPlug("vertexMatrix"), fdeformer.findPlug("vertexMatrix"));
	MStatus stat = mod.doIt();
	
/// find  input[meshId].inputGeometry
	MPlug inputPlug = fdeformer.findPlug("input");
	MPlug ainputPlug = inputPlug.elementByLogicalIndex(meshId);

/// child(0) is inputGeometry
	MPlug geoPlug = ainputPlug.child(0);
	MPlugArray srcs;
	if(!geoPlug.connectedTo (srcs, true, false)) {
		AHelper::Info<MString>("sticky error no connection ", geoPlug.name() );
		return;
	}
	
	AHelper::Info<MString>("sticky deformer input geometry", srcs[0].name() );

	mod.connect(srcs[0], fviz.findPlug("inMesh"));
	stat = mod.doIt();
	if(!stat) {
		AHelper::Info<MString>("sticky error cannot connect viz ", srcs[0].name() );
		return;
	}
}

void StickyCmd::storeRef(const MObject & viz, const MDagPath & mesh, unsigned & ivert)
{
	MItMeshVertex itmv(mesh);
	int prevIndex;
	itmv.setIndex (ivert , prevIndex);
	MIntArray vertexList;
	itmv.getConnectedVertices ( vertexList );
	AHelper::Info<MIntArray>("neighbors", vertexList);
	
	MFnDependencyNode fviz(viz);
	
	MFnIntArrayData indexD;
	MObject oindex = indexD.create(vertexList);
	
	MPlug indexPlug = fviz.findPlug("refInds");
	indexPlug.setValue(oindex);
	
	MFnMesh fmesh(mesh);
	MPoint cen, pt;
	fmesh.getPoint(ivert, cen);
	const unsigned nv = vertexList.length();
	MVectorArray dvs;
	unsigned i=0;
	for(;i<nv;++i) {
		fmesh.getPoint(vertexList[i], pt);
		dvs.append(pt - cen);
	}
	
	AHelper::Info<MVectorArray>("dv", dvs);
	
	MFnVectorArrayData displaceD;
	MObject odisplace = displaceD.create(dvs);
	
	MPlug displacePlug = fviz.findPlug("refDisplace");
	displacePlug.setValue(odisplace);
}
//:~