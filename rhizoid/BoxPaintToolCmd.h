#ifndef BOXPAINTTOOLCMD_H
#define BOXPAINTTOOLCMD_H

/*
 *  BoxPaintToolCmd.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MString.h>
#include <maya/MGlobal.h>
#include <maya/M3dView.h>
#include <maya/MDagPath.h>
 
#include <maya/MItSelectionList.h>
#include <maya/MSelectionList.h>

#include <maya/MPxContextCommand.h>
#include <maya/MPxContext.h>
#include <maya/MPxToolCommand.h> 
#include <maya/MEvent.h>
#include <maya/MToolsInfo.h>

#include <maya/MSyntax.h>
#include <maya/MArgParser.h>
#include <maya/MArgDatabase.h>
#include <maya/MArgList.h>

#include <maya/MDagPathArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MMatrix.h>
#include <maya/MFnMesh.h>

#if defined(OSMac_MachO_)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <Matrix44F.h>
#include "ProxyVizNode.h"
#include <PseudoNoise.h>
#include <IntersectionGroup.h>

#define kOptFlag "-opt" 
#define kOptFlagLong "-option"
#define kNsegFlag "-nsg" 
#define kNsegFlagLong "-numSegment"
#define kLsegFlag "-brd" 
#define kLsegFlagLong "-brushRadius"
#define kMinFlag "-smn" 
#define kMinFlagLong "-scaleMin"
#define kMaxFlag "-smx" 
#define kMaxFlagLong "-scaleMax"
#define kRotateNoiseFlag "-rno" 
#define kRotateNoiseFlagLong "-rotateNoise"
#define kWeightFlag "-bwt" 
#define kWeightFlagLong "-brushWeight"
#define kNormalFlag "-anl" 
#define kNormalFlagLong "-alongNormal"
#define kWriteCacheFlag "-wch" 
#define kWriteCacheFlagLong "-writeCache"
#define kReadCacheFlag "-rch" 
#define kReadCacheFlagLong "-readCache"
#define kCullSelectionFlag "-cus" 
#define kCullSelectionFlagLong "-cullSelection"
#define kMultiCreateFlag "-mct" 
#define kMultiCreateFlagLong "-multiCreate"
#define kInstanceGroupCountFlag "-igc" 
#define kInstanceGroupCountFlagLong "-instanceGroupCount"
#define kBlockFlag "-bl" 
#define kBlockFlagLong "-block"
#define kVizFlag "-v" 
#define kVizFlagLong "-visualizer"

class proxyPaintTool : public MPxToolCommand
{
public:
					proxyPaintTool(); 
	virtual			~proxyPaintTool(); 
	static void*	creator();

	MStatus			doIt(const MArgList& args);
	MStatus			parseArgs(const MArgList& args);
	static MSyntax	newSyntax();
	MStatus			finalize();

private:
	MStatus connectSelected();
	bool connectMeshToViz(MObject & meshObj, MObject & vizObj, unsigned & slot);
	void connectTransform(MObject & transObj, MObject & vizObj, const unsigned & slot);
	MStatus saveCacheSelected();
	MStatus loadCacheSelected();
	MObject getSelectedViz(const MSelectionList & sels, MStatus & stat);
	
private:
	enum Operation {
		opUnknown = 0,
		opBeginPick = 1,
		opDoPick = 2,
		opEndPick = 3,
		opGetPick = 4,
		opConnectGround = 5,
		opSaveCache = 6,
		opLoadCache = 7
	};
	
	Operation m_operation;
	
	unsigned opt, nseg;
	float lseg;
	MString fBlockerName, fVizName, m_cacheName;
};

const char helpString[] =
			"Select a proxy viz and a ground plane to paint on.";
#endif        //  #ifndef BOXPAINTTOOLCMD_H

