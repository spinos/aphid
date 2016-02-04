/*
 *  BoxPaintToolCmd.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoxPaintToolCmd.h"
#include "ProxyVizNode.h"
#include <ASearchHelper.h>

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

	syntax.addFlag(kOptFlag, kOptFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kNsegFlag, kNsegFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kWeightFlag, kWeightFlagLong, MSyntax::kDouble );
	syntax.addFlag(kLsegFlag, kLsegFlagLong, MSyntax::kDouble );
	syntax.addFlag(kMinFlag, kMinFlagLong, MSyntax::kDouble );
	syntax.addFlag(kMaxFlag, kMaxFlagLong, MSyntax::kDouble );
	syntax.addFlag(kRotateNoiseFlag, kRotateNoiseFlagLong, MSyntax::kDouble );
	syntax.addFlag(kNormalFlag, kNormalFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kWriteCacheFlag, kWriteCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kReadCacheFlag, kReadCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kCullSelectionFlag, kCullSelectionFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kInstanceGroupCountFlag, kInstanceGroupCountFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kBeginPickFlag, kBeginPickFlagLong, MSyntax::kString);
	syntax.addFlag(kDoPickFlag, kDoPickFlagLong, MSyntax::kString);
	syntax.addFlag(kEndPickFlag, kEndPickFlagLong, MSyntax::kString);
	syntax.addFlag(kEndPickFlag, kEndPickFlagLong, MSyntax::kString);
	syntax.addFlag(kGetPickFlag, kGetPickFlagLong, MSyntax::kString);
	syntax.addFlag(kSaveCacheFlag, kSaveCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kLoadCacheFlag, kLoadCacheFlagLong, MSyntax::kString);
	syntax.addFlag(kConnectGroundFlag, kConnectGroundFlagLong);
	
	return syntax;
}

MStatus proxyPaintTool::doIt(const MArgList &args)
//
// Description
//     Sets up the helix parameters from arguments passed to the
//     MEL command.
//
{
	MStatus status;

	status = parseArgs(args);
	
	if(m_operation == opUnknown) return status;
	
	if(m_operation == opConnectGround) return connectSelected();
	
	if(m_operation == opSaveCache) return saveCacheSelected();
			
	if(m_operation == opLoadCache) return loadCacheSelected();
		
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
	
	if (argData.isFlagSet(kOptFlag)) {
		unsigned tmp;
		status = argData.getFlagArgument(kOptFlag, 0, tmp);
		if (!status) {
			status.perror("opt flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kNsegFlag)) {
		unsigned tmp;
		status = argData.getFlagArgument(kNsegFlag, 0, tmp);
		if (!status) {
			status.perror("numseg flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kLsegFlag)) {
		double tmp;
		status = argData.getFlagArgument(kLsegFlag, 0, tmp);
		if (!status) {
			status.perror("lenseg flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kWeightFlag)) {
		double tmp;
		status = argData.getFlagArgument(kWeightFlag, 0, tmp);
		if (!status) {
			status.perror("weight flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kNormalFlag)) {
		unsigned aln;
		status = argData.getFlagArgument(kNormalFlag, 0, aln);
		if (!status) {
			status.perror("normal flag parsing failed");
			return status;
		}
	}

	if (argData.isFlagSet(kWriteCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kWriteCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache out flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kReadCacheFlag)) 
	{
		MString ch;
		status = argData.getFlagArgument(kReadCacheFlag, 0, ch);
		if (!status) {
			status.perror("cache in flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kMinFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMinFlag, 0, noi);
		if (!status) {
			status.perror("min flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kMaxFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kMaxFlag, 0, noi);
		if (!status) {
			status.perror("max flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kRotateNoiseFlag)) 
	{
		double noi;
		status = argData.getFlagArgument(kRotateNoiseFlag, 0, noi);
		if (!status) {
			status.perror("rotate noise flag parsing failed.");
			return status;
		}
	}
	
	if (argData.isFlagSet(kCullSelectionFlag)) {
		unsigned cus;
		status = argData.getFlagArgument(kCullSelectionFlag, 0, cus);
		if (!status) {
			status.perror("cull selection flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kInstanceGroupCountFlag)) {
		unsigned igc;
		status = argData.getFlagArgument(kInstanceGroupCountFlag, 0, igc);
		if (!status) {
			status.perror("instance group count flag parsing failed");
			return status;
		}
	}
	
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
//
// Description
//     Command is finished, construct a string for the command
//     for journalling.
//
{
	MArgList command;
	command.addArg(commandString());
	//command.addArg(MString(kOptFlag));
	//command.addArg((int)opt);
	//command.addArg(MString(kNSegFlag));
	//command.addArg((int)nseg);
	//command.addArg(MString(kLSegFlag));
	//command.addArg((float)lseg);
	return MPxToolCommand::doFinalize( command );
}

MStatus proxyPaintTool::connectSelected()
{
	MSelectionList sels;
 	MGlobal::getActiveSelectionList( sels );
	
	if(sels.length() < 2) {
		MGlobal::displayWarning("proxyPaintTool wrong selection, select mesh(es) and a viz to connect");
		return MS::kFailure;
	}
	
	MStatus stat;
	MObject vizobj = getSelectedViz(sels, stat);
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
	
	MObject vizobj = getSelectedViz(sels, stat);
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
	
	MObject vizobj = getSelectedViz(sels, stat);
	if(!stat) return stat;
	
	MFnDependencyNode fviz(vizobj, &stat);
	AHelper::Info<MString>("proxyPaintTool found viz node", fviz.name() );
		
	ProxyViz* pViz = (ProxyViz*)fviz.userNode();
	pViz->loadExternal(m_cacheName.asChar() );
	
	return stat;
}

MObject proxyPaintTool::getSelectedViz(const MSelectionList & sels, MStatus & stat)
{
	MItSelectionList iter(sels, MFn::kPluginLocatorNode, &stat );
    MObject vizobj;
    iter.getDependNode(vizobj);
	MFnDependencyNode fviz(vizobj, &stat);
    if(stat) {
        MFnDependencyNode fviz(vizobj);
		if(fviz.typeName() != "proxyViz") stat = MS::kFailure;
	}
	
	if(!stat )
		AHelper::Info<int>("proxyPaintTool error no viz node selected", 0);
		
	return vizobj;
}	
//:~