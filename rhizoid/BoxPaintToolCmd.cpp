/*
 *  BoxPaintToolCmd.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoxPaintToolCmd.h"
#include <maya/MDagModifier.h>
#include "ProxyVizNode.h"
#include "ExampVizNode.h"
#include <ASearchHelper.h>
#include <ATriangleMesh.h>
#include <KdIntersection.h>

#define kBeginPickFlag "-bpk" 
#define kBeginPickFlagLong "-beginPick"
#define kDoPickFlag "-dpk" 
#define kDoPickFlagLong "-doPick"
#define kEndPickFlag "-epk" 
#define kEndPickFlagLong "-endPick"
#define kGetPickFlag "-gpk" 
#define kGetPickFlagLong "-getPick"
#define kConnectGroundFlag "-cgm" 
#define kConnectGroundFlagLong "-connectGMesh"
#define kSaveCacheFlag "-scf" 
#define kSaveCacheFlagLong "-saveCacheFile"
#define kLoadCacheFlag "-lcf" 
#define kLoadCacheFlagLong "-loadCacheFile"
#define kVoxFlag "-vxl" 
#define kVoxFlagLong "-voxelize"
#define kConnectVoxFlag "-cvx" 
#define kConnectVoxFlagLong "-connectVoxel"

proxyPaintTool::~proxyPaintTool() {}

proxyPaintTool::proxyPaintTool()
{
	setCommandString("proxyPaintToolCmd");
}

void* proxyPaintTool::creator()
{
	return new proxyPaintTool;
}

MSyntax proxyPaintTool::newSyntax()
{
	MSyntax syntax;

	syntax.addFlag(kBeginPickFlag, kBeginPickFlagLong, MSyntax::kString);
	syntax.addFlag(kDoPickFlag, kDoPickFlagLong, MSyntax::kString);
	syntax.addFlag(kEndPickFlag, kEndPickFlagLong, MSyntax::kString);
	syntax.addFlag(kEndPickFlag, kEndPickFlagLong, MSyntax::kString);
	syntax.addFlag(kGetPickFlag, kGetPickFlagLong, MSyntax::kString);
	syntax.addFlag(kSaveCacheFlag, kSaveCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kLoadCacheFlag, kLoadCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kConnectGroundFlag, kConnectGroundFlagLong);
	syntax.addFlag(kVoxFlag, kVoxFlagLong);
	syntax.addFlag(kConnectVoxFlag, kConnectVoxFlagLong);
	
	return syntax;
}

MStatus proxyPaintTool::doIt(const MArgList &args)
{
	MStatus status;

	status = parseArgs(args);
	
	if(m_operation == opUnknown) return status;
	
	if(m_operation == opConnectGround) return connectGroundSelected();
	
	if(m_operation == opSaveCache) return saveCacheSelected();
			
	if(m_operation == opLoadCache) return loadCacheSelected();
	
	if(m_operation == opVoxelize) return voxelizeSelected();
	
	if(m_operation == opConnectVoxel) return connectVoxelSelected();
		
	ASearchHelper finder;

	MObject oViz;
	if(!finder.getObjByFullName(fVizName.asChar(), oViz)) {
		MGlobal::displayWarning(MString("cannot find viz: ") + fVizName);
		return MS::kSuccess;
	}
	
	MFnDependencyNode fviz(oViz);
    ProxyViz *pViz = (ProxyViz*)fviz.userNode();
    
    if(!pViz) {
		MGlobal::displayWarning(MString("cannot recognize viz: ") + fVizName);
		return MS::kSuccess;
	}
	
	switch(m_operation) {
		case opBeginPick:
			pViz->beginPickInView();
			break;
		case opDoPick:
			pViz->processPickInView();
			break;
		case opEndPick:
			pViz->endPickInView();
			break;
		case opGetPick:
			setResult((int)pViz->numActivePlants() );
			break;
		default:
			;
	}

	return MS::kSuccess;
}

MStatus proxyPaintTool::parseArgs(const MArgList &args)
{
	m_operation = opUnknown;
	MStatus status;
	MArgDatabase argData(syntax(), args);

	if (argData.isFlagSet(kBeginPickFlag)) {
		status = argData.getFlagArgument(kBeginPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opBeginPick;
	}
	
	if (argData.isFlagSet(kDoPickFlag)) {
		status = argData.getFlagArgument(kDoPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opDoPick;
	}
	
	if (argData.isFlagSet(kEndPickFlag)) {
		status = argData.getFlagArgument(kEndPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opEndPick;
	}
	
	if (argData.isFlagSet(kGetPickFlag)) {
		status = argData.getFlagArgument(kGetPickFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
		m_operation = opGetPick;
	}
	
	if (argData.isFlagSet(kConnectGroundFlag))
		m_operation = opConnectGround;
	
	if (argData.isFlagSet(kVoxFlag))
		m_operation = opVoxelize;
		
	if (argData.isFlagSet(kConnectVoxFlag))
		m_operation = opConnectVoxel;
	
	if (argData.isFlagSet(kSaveCacheFlag)) {
		status = argData.getFlagArgument(kSaveCacheFlag, 0, m_cacheName);
		if (!status) {
			status.perror("save cache flag parsing failed");
			return status;
		}
		m_operation = opSaveCache;
	}
	
	if (argData.isFlagSet(kLoadCacheFlag)) {
		status = argData.getFlagArgument(kLoadCacheFlag, 0, m_cacheName);
		if (!status) {
			status.perror("save cache flag parsing failed");
			return status;
		}
		m_operation = opLoadCache;
	}
	
	return MS::kSuccess;
}

MStatus proxyPaintTool::finalize()
{
	MArgList command;
	command.addArg(commandString());
	return MPxToolCommand::doFinalize( command );
}

MStatus proxyPaintTool::connectGroundSelected()
{
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 2) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) and a viz to connect");
		return MS::kFailure;
	}
	
	MStatus stat;
	MObject vizobj = getSelectedViz(sels, "proxyViz", stat);
	if(!stat) return stat;
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->setEnableCompute(false);
	
	MItSelectionList meshIter(sels, MFn::kMesh, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintTool no mesh selected");
		return MS::kFailure;
	}
	
	for(;!meshIter.isDone(); meshIter.next() ) {
		MObject mesh;
		meshIter.getDependNode(mesh);
		
		MFnDependencyNode fmesh(mesh, &stat);
		if(!stat) continue;
			
		AHelper::Info<MString>("proxyPaintTool found mesh", fmesh.name() );
		unsigned islot;
		if(connectMeshToViz(mesh, vizobj, islot)) {
			MDagPath meshPath;
			meshIter.getDagPath ( meshPath );
			meshPath.pop();
			AHelper::Info<MString>("proxyPaintTool connect transform", meshPath.fullPathName() );
			MObject trans = meshPath.node();
			connectTransform(trans, vizobj, islot);
		}
	}
	
	pViz->setEnableCompute(true);
	
	return stat;
}

MStatus proxyPaintTool::connectVoxelSelected()
{
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 2) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select proxyExample(s) and a viz to connect");
		return MS::kFailure;
	}
	
	MStatus stat;
	MObject vizobj = getSelectedViz(sels, "proxyViz", stat);
	if(!stat) return stat;
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
	
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->setEnableCompute(false);
	
	MItSelectionList voxIter(sels, MFn::kPluginLocatorNode, &stat);
	
	for(;!voxIter.isDone(); voxIter.next() ) {
		MObject vox;
		voxIter.getDependNode(vox);
		
		MFnDependencyNode fvox(vox, &stat);
		if(!stat) continue;
		
		if(fviz.typeName() != "proxyExample") continue;
			
		AHelper::Info<MString>("proxyPaintTool found proxyExample", fvox.name() );
		unsigned islot;
		if(connectVoxToViz(vox, vizobj, islot) )
			checkOutputConnection(vizobj, "ov1");
	}
	
	pViz->setEnableCompute(true);
	
	return stat;
}

bool proxyPaintTool::connectMeshToViz(MObject & meshObj, MObject & vizObj, unsigned & slot)
{
	MFnDependencyNode fmesh(meshObj);
	MPlug srcMesh = fmesh.findPlug("outMesh");
	AHelper::Info<MString>("check", srcMesh.name() );
	
	MStatus stat;
	MPlugArray connected;
	srcMesh.connectedTo ( connected , false, true, &stat );
	unsigned i = 0;
	for(;i<connected.length();++i) {
		if(connected[i].node() == vizObj) {
			AHelper::Info<MString>("already connected to", connected[i].name() );
			return false;
		}
	}
	
	MFnDependencyNode fviz(vizObj);
	MPlug dstGround = fviz.findPlug("groundMesh");
	slot = dstGround.numElements();
	MPlug dst = dstGround.elementByLogicalIndex(slot);
	AHelper::Info<MString>("connect to", dst.name() );
	
	MDGModifier modif;
	modif.connect(srcMesh, dst );
	modif.doIt();
	return true;
}

bool proxyPaintTool::connectVoxToViz(MObject & voxObj, MObject & vizObj, unsigned & slot)
{
	MFnDependencyNode fvox(voxObj);
	MPlug srcPlug = fvox.findPlug("ov");
	AHelper::Info<MString>("check", srcPlug.name() );
	
	MStatus stat;
	MPlugArray connected;
	srcPlug.connectedTo ( connected , false, true, &stat );
	unsigned i = 0;
	for(;i<connected.length();++i) {
		if(connected[i].node() == vizObj) {
			AHelper::Info<MString>("already connected to", connected[i].name() );
			return false;
		}
	}
	
	MFnDependencyNode fviz(vizObj);
	MPlug inExample = fviz.findPlug("ixmp");
	slot = inExample.numElements();
	MPlug dstPlug = inExample.elementByLogicalIndex(slot);
	AHelper::Info<MString>("connect to", dstPlug.name() );
	
	MDGModifier modif;
	modif.connect(srcPlug, dstPlug );
	modif.doIt();
	return true;
}

void proxyPaintTool::connectTransform(MObject & transObj, MObject & vizObj, const unsigned & slot)
{
	MFnDependencyNode ftrans(transObj);
	MPlug srcSpace = ftrans.findPlug("worldMatrix").elementByLogicalIndex(0);
	AHelper::Info<MString>("check mesh", srcSpace.name() );

	MFnDependencyNode fviz(vizObj);
	MPlug dstGround = fviz.findPlug("groundSpace");
	MPlug dst = dstGround.elementByLogicalIndex(slot);
	AHelper::Info<MString>("connect to", dst.name() );
	
	MDGModifier modif;
	modif.connect(srcSpace, dst );
	modif.doIt();
}

MStatus proxyPaintTool::saveCacheSelected()
{
	MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select a viz to save cache");
		return MS::kFailure;
	}
	
	MObject vizobj = getSelectedViz(sels, "proxyViz", stat);
	if(!stat) return stat;
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->saveExternal(m_cacheName.asChar() );
	
	return stat;
}
	
MStatus proxyPaintTool::loadCacheSelected()
{
	MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select a viz to load cache");
		return MS::kFailure;
	}
	
	MObject vizobj = getSelectedViz(sels, "proxyViz", stat);
	if(!stat) return stat;
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->loadExternal(m_cacheName.asChar() );
	
	return stat;
}

// "proxyViz" or "proxyExample"
MObject proxyPaintTool::getSelectedViz(const MSelectionList & sels, 
									const MString & typName,
									MStatus & stat)
{
	MItSelectionList iter(sels, MFn::kPluginLocatorNode, &stat );
    MObject vizobj;
    iter.getDependNode(vizobj);
	MFnDependencyNode fviz(vizobj, &stat);
    if(stat) {
        MFnDependencyNode fviz(vizobj);
		if(fviz.typeName() != typName) stat = MS::kFailure;
	}
	
	if(!stat )
		AHelper::Info<int>("proxyPaintTool error no viz node selected", 0);
		
	return vizobj;
}

MStatus proxyPaintTool::voxelizeSelected()
{
	MStatus stat;
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 1) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) to voxelize");
		return MS::kFailure;
	}
	
	MObject vizobj = getSelectedViz(sels, "proxyExample", stat);
	if(!stat) {
		vizobj = createViz("proxyExample", "proxyExample");
		if(vizobj.isNull()) {
			MGlobal::displayWarning("proxyPaintTool cannot create example viz");
			return MS::kFailure;
		}
	}
	
	MItSelectionList meshIter(sels, MFn::kMesh, &stat);
	if(!stat) {
		MGlobal::displayWarning("proxyPaintTool no mesh selected");
		return MS::kFailure;
	}
	
	std::vector<ATriangleMesh * > meshes;
	for(;!meshIter.isDone(); meshIter.next() ) {
		MObject mesh;
		meshIter.getDependNode(mesh);
		
		MFnMesh fmesh(mesh, &stat);
		if(!stat) continue;
			
		AHelper::Info<MString>("proxyPaintTool voxelize add mesh", fmesh.name() );
	
		MDagPath meshPath;
		meshIter.getDagPath(meshPath);
		
		MMatrix wm = AHelper::GetWorldTransformMatrix(meshPath);
		
		MPointArray ps;
		fmesh.getPoints(ps);
		
		const unsigned nv = ps.length();
		unsigned i = 0;
		for(;i<nv;i++) ps[i] *= wm;
		
		MIntArray triangleCounts, triangleVertices;
		fmesh.getTriangles(triangleCounts, triangleVertices);
		
		ATriangleMesh * trimesh = new ATriangleMesh;
		trimesh->create(nv, triangleVertices.length()/3);
		
		Vector3F * cvs = trimesh->points();
		unsigned * ind = trimesh->indices();
		for(i=0;i<nv;i++) cvs[i].set(ps[i].x, ps[i].y, ps[i].z);
		for(i=0;i<triangleVertices.length();i++) ind[i] = triangleVertices[i];
		
		meshes.push_back(trimesh);
	}
	
	if(meshes.size() < 1) {
		MGlobal::displayWarning("proxyPaintTool no mesh added");
		return MS::kFailure;
	}
	
	AHelper::Info<unsigned>("proxyPaintTool voxelize n mesh", meshes.size() );
	
	KdTree::MaxBuildLevel = 24;
	KdTree::NumPrimitivesInLeafThreashold = 24;
	
	KdIntersection tree;
	std::vector<ATriangleMesh * >::iterator it = meshes.begin();
	for(;it!= meshes.end();++it)
		tree.addGeometry(*it);
		
	tree.create();
	
	AHelper::Info<BoundingBox>("proxyPaintTool bounding", tree.getBBox());
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool init viz node", fviz.name() );
		
	ExampViz* pViz = (ExampViz*)fviz.userNode();
	pViz->voxelize(&tree );
	
	it = meshes.begin();
	for(;it!= meshes.end();++it) delete *it;
	meshes.clear();
	return stat;
}

MObject proxyPaintTool::createViz(const MString & typName,
									const MString & transName)
{
	MDagModifier modif;
	MObject trans = modif.createNode("transform");
	modif.renameNode (trans, transName);
	MObject viz = modif.createNode(typName, trans);
	modif.doIt();
	MString vizName = MFnDependencyNode(trans).name() + "Shape";
	modif.renameNode(viz, vizName);
	modif.doIt();
	return viz;
}

void proxyPaintTool::checkOutputConnection(MObject & node, const MString & outName)
{
}
//:~