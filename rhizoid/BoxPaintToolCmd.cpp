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
#include <HesperisIO.h>

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
#define kSelectVoxFlag "-svx" 
#define kSelectVoxFlagLong "-selectVox"

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
	syntax.addFlag(kSelectVoxFlag, kSelectVoxFlagLong, MSyntax::kLong);
	
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
		
	aphid::ASearchHelper finder;

	MObject oViz;
	if(!finder.getObjByFullName(fVizName.asChar(), oViz)) {
		MGlobal::displayWarning(MString("cannot find viz: ") + fVizName);
		return MS::kSuccess;
	}
	
	MFnDependencyNode fviz(oViz);
    aphid::ProxyViz *pViz = (aphid::ProxyViz*)fviz.userNode();
    
    if(!pViz) {
		MGlobal::displayWarning(MString("cannot recognize viz: ") + fVizName);
		return MS::kSuccess;
	}
	
	switch(m_operation) {
		case opBeginPick:
			pViz->beginPickInView();
			break;
		case opDoPick:
			pViz->processPickInView(m_currentVoxInd);
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
    m_currentVoxInd = 0;
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
    
    if (argData.isFlagSet(kSelectVoxFlag)) {
		status = argData.getFlagArgument(kSelectVoxFlag, 0, m_currentVoxInd);
		if (!status) {
			status.perror("-selectVox flag parsing failed");
			return status;
		}
		aphid::AHelper::Info<int>(" proxyPaintTool select example", m_currentVoxInd);
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
	aphid::AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	aphid::ProxyViz* pViz = (aphid::ProxyViz*)fviz.userNode();
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
			
		unsigned islot;
		if(connectMeshToViz(mesh, vizobj, islot)) {
			MDagPath meshPath;
			meshIter.getDagPath ( meshPath );
			meshPath.pop();
			aphid::AHelper::Info<MString>("proxyPaintTool connect ground", meshPath.fullPathName() );
			MObject trans = meshPath.node();
			connectTransform(trans, vizobj, islot);
			checkOutputConnection(vizobj, "ov");
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
	aphid::AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
	
	aphid::ProxyViz* pViz = (aphid::ProxyViz*)fviz.userNode();
	pViz->setEnableCompute(false);
	
	MItSelectionList voxIter(sels, MFn::kPluginLocatorNode, &stat);
	
	for(;!voxIter.isDone(); voxIter.next() ) {
		MObject vox;
		voxIter.getDependNode(vox);
		
		MFnDependencyNode fvox(vox, &stat);
		if(!stat) continue;
		
		if(fvox.typeName() != "proxyExample") continue;
			
		unsigned islot;
		if(connectVoxToViz(vox, vizobj, islot) ) {
			aphid::AHelper::Info<MString>("proxyPaintTool connect example", fvox.name() );
			checkOutputConnection(vizobj, "ov1");
		}
	}
	
	pViz->setEnableCompute(true);
	
	return stat;
}

bool proxyPaintTool::connectMeshToViz(MObject & meshObj, MObject & vizObj, unsigned & slot)
{
	MFnDependencyNode fmesh(meshObj);
	MPlug srcMesh = fmesh.findPlug("outMesh");
	aphid::AHelper::Info<MString>("check", srcMesh.name() );
	
	MStatus stat;
	MPlugArray connected;
	srcMesh.connectedTo ( connected , false, true, &stat );
	unsigned i = 0;
	for(;i<connected.length();++i) {
		if(connected[i].node() == vizObj) {
			aphid::AHelper::Info<MString>("already connected to", connected[i].name() );
			return false;
		}
	}
	
	MFnDependencyNode fviz(vizObj);
	MPlug dstGround = fviz.findPlug("groundMesh");
	slot = dstGround.numElements();
	MPlug dst = dstGround.elementByLogicalIndex(slot);
	aphid::AHelper::Info<MString>("connect to", dst.name() );
	
	MDGModifier modif;
	modif.connect(srcMesh, dst );
	modif.doIt();
	return true;
}

bool proxyPaintTool::connectVoxToViz(MObject & voxObj, MObject & vizObj, unsigned & slot)
{
	MFnDependencyNode fvox(voxObj);
	MPlug srcPlug = fvox.findPlug("ov");
	aphid::AHelper::Info<MString>("check", srcPlug.name() );
	
	MStatus stat;
	MPlugArray connected;
	srcPlug.connectedTo ( connected , false, true, &stat );
	unsigned i = 0;
	for(;i<connected.length();++i) {
		if(connected[i].node() == vizObj) {
			aphid::AHelper::Info<MString>("already connected to", connected[i].name() );
			return false;
		}
	}
	
	MFnDependencyNode fviz(vizObj);
	MPlug inExample = fviz.findPlug("ixmp");
	slot = inExample.numElements();
	MPlug dstPlug = inExample.elementByLogicalIndex(slot);
	aphid::AHelper::Info<MString>("connect to", dstPlug.name() );
	
	MDGModifier modif;
	modif.connect(srcPlug, dstPlug );
	modif.doIt();
	return true;
}

void proxyPaintTool::connectTransform(MObject & transObj, MObject & vizObj, const unsigned & slot)
{
	MFnDependencyNode ftrans(transObj);
	MPlug srcSpace = ftrans.findPlug("worldMatrix").elementByLogicalIndex(0);
	aphid::AHelper::Info<MString>("check mesh", srcSpace.name() );

	MFnDependencyNode fviz(vizObj);
	MPlug dstGround = fviz.findPlug("groundSpace");
	MPlug dst = dstGround.elementByLogicalIndex(slot);
	aphid::AHelper::Info<MString>("connect to", dst.name() );
	
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
	aphid::AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	aphid::ProxyViz* pViz = (aphid::ProxyViz*)fviz.userNode();
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
	aphid::AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	aphid::ProxyViz* pViz = (aphid::ProxyViz*)fviz.userNode();
	pViz->loadExternal(m_cacheName.asChar() );
	
	return stat;
}

MObject proxyPaintTool::getSelectedViz(const MSelectionList & sels, 
									const MString & typName,
									MStatus & stat)
{
	MItSelectionList iter(sels, MFn::kPluginLocatorNode, &stat );
	stat = MS::kFailure;
	MObject vizobj;
	for(;!iter.isDone();iter.next() ) {
		iter.getDependNode(vizobj);
		MFnDependencyNode fviz(vizobj, &stat);
		if(stat) {
			MFnDependencyNode fviz(vizobj);
			if(fviz.typeName() == typName) {
				stat = MS::kSuccess;
				break;
			}
		}
	}
	
	if(!stat )
		aphid::AHelper::Info<MString>("proxyPaintTool select no node by type", typName);
		
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
	
	aphid::sdb::VectorArray<aphid::cvx::Triangle> tris;
	aphid::BoundingBox bbox;
	
	for(;!meshIter.isDone(); meshIter.next() ) {
		
		MDagPath meshPath;
		meshIter.getDagPath(meshPath);
		getMeshTris(tris, bbox, meshPath);
	}
	
	if(tris.size() < 1) {
		MGlobal::displayWarning("proxyPaintTool no triangle added");
		return MS::kFailure;
	}
	
	bbox.round();
	aphid::AHelper::Info<unsigned>("proxyPaintTool voxelize n triangle", tris.size() );

	MFnDependencyNode fviz(vizobj, &stat);
	aphid::AHelper::Info<MString>("proxyPaintTool init viz node", fviz.name() );
		
	ExampViz* pViz = (ExampViz*)fviz.userNode();
	pViz->voxelize1(&tris, bbox);
	
	return stat;
}

void proxyPaintTool::getMeshTris(aphid::sdb::VectorArray<aphid::cvx::Triangle> & tris,
								aphid::BoundingBox & bbox,
								const MDagPath & meshPath)
{
	aphid::AHelper::Info<MString>("get mesh triangles", meshPath.fullPathName() );
	
	MMatrix worldTm = aphid::HesperisIO::GetWorldTransform(meshPath);
	
    MStatus stat;
	
    MIntArray vertices;
    int i, j, nv;
	MPoint dp[3];
	aphid::Vector3F fp[3];
	MItMeshPolygon faceIt(meshPath);
    for(; !faceIt.isDone(); faceIt.next() ) {

		faceIt.getVertices(vertices);
        nv = vertices.length();
        
        for(i=1; i<nv-1; ++i ) {
			dp[0] = faceIt.point(0, MSpace::kObject );
			dp[1] = faceIt.point(i, MSpace::kObject );
			dp[2] = faceIt.point(i+1, MSpace::kObject );
			
			dp[0] *= worldTm;
			dp[1] *= worldTm;	
			dp[2] *= worldTm;
			
			aphid::cvx::Triangle tri;
			for(j=0; j<3; ++j) {
				fp[j].set(dp[j].x, dp[j].y, dp[j].z);
				tri.setP(fp[j], j);
				bbox.expandBy(fp[j], 1e-4f);
			}
			
			tris.insert(tri);
        }
    }
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
	MFnDependencyNode fnode(node);
	MStatus stat;
	MPlug outPlug = fnode.findPlug(outName, &stat);
	if(!stat) {
		aphid::AHelper::Info<MString>(" proxyPaintTool error not named plug", outName);
		return;
	}
	
	if(outPlug.isConnected() ) return;
	
	MDagModifier modif;
	MObject trans = modif.createNode("transform");
	modif.renameNode (trans, fnode.name() + "_" + outName);
	modif.doIt();
	
	modif.connect(outPlug, MFnDependencyNode(trans).findPlug("tx") );
	modif.doIt();
	
	MFnDagNode fdag(node);
	MDagPath parentPath;
	fdag.getPath(parentPath);
	parentPath.pop();
	
	modif.reparentNode(trans, parentPath.node() );
	modif.doIt();
}
//:~