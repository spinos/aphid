/*
 *  BoxPaintToolCmd.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "BoxPaintToolCmd.h"
#include <ASearchHelper.h>

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
	syntax.addFlag(kMultiCreateFlag, kMultiCreateFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kInstanceGroupCountFlag, kInstanceGroupCountFlagLong, MSyntax::kUnsigned);
	syntax.addFlag(kBlockFlag, kBlockFlagLong, MSyntax::kString);
	syntax.addFlag(kVizFlag, kVizFlagLong, MSyntax::kString);
	
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

	return MS::kSuccess;
}

MStatus proxyPaintTool::parseArgs(const MArgList &args)
{
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
	
	if (argData.isFlagSet(kMultiCreateFlag)) {
		unsigned mcr;
		status = argData.getFlagArgument(kMultiCreateFlag, 0, mcr);
		if (!status) {
			status.perror("multi create flag parsing failed");
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
	
	if (argData.isFlagSet(kBlockFlag)) {
		status = argData.getFlagArgument(kBlockFlag, 0, fBlockerName);
		if (!status) {
			status.perror("block flag parsing failed");
			return status;
		}
	}
	
	if (argData.isFlagSet(kVizFlag)) {
		status = argData.getFlagArgument(kVizFlag, 0, fVizName);
		if (!status) {
			status.perror("viz flag parsing failed");
			return status;
		}
	}
	
	MGlobal::displayInfo(MString("culled by ") + fBlockerName + " " + fVizName);
	
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