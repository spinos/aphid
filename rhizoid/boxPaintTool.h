/*
 *  proxyPaintTool.h
 *  hair
 *
 *  Created by jian zhang on 6/3/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
#include <math.h>
#include <stdlib.h>

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
#include "proxyVizNode.h"
#include <PseudoNoise.h>
#include <IntersectionGroup.h>

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
	unsigned opt, nseg;
	float lseg;
	MString fBlockerName, fVizName;
};

const char helpString[] =
			"Select a proxy viz and a ground plane to paint on.";

class proxyPaintContext : public MPxContext
{
	float m_brushRadius, m_brushWeight;
	float m_min_scale, m_max_scale, m_rotation_noise;
	unsigned m_growAlongNormal;
	
public:
					proxyPaintContext();
	virtual void	toolOnSetup( MEvent & event );
	virtual MStatus	doPress( MEvent & event );
	virtual MStatus	doDrag( MEvent & event );
	virtual MStatus	doRelease( MEvent & event );
	virtual MStatus	doEnterRegion( MEvent & event );
	
	virtual	void	getClassName(MString & name) const;
	
	void setOperation(unsigned val);
	unsigned getOperation() const;
	void setNSegment(unsigned val);
	unsigned getNSegment() const;
	void setBrushRadius(float val);
	float getBrushRadius() const;
	void setScaleMin(float val);
	float getScaleMin() const;
	void setScaleMax(float val);
	float getScaleMax() const;
	void setRotationNoise(float val);
	float getRotationNoise() const;
	void setBrushWeight(float val);
	float getBrushWeight() const;
	void setGrowAlongNormal(unsigned val);
	unsigned getGrowAlongNormal() const;
	void setCullSelection(unsigned val);
	unsigned getCullSelection() const;
	void setMultiCreate(unsigned val);
	unsigned getMultiCreate() const;
	void setInstanceGroupCount(unsigned val);
	unsigned getInstanceGroupCount() const;
	void setWriteCache(MString filename);
	void setReadCache(MString filename);
	void cleanup();
	char getSelectedViz();

private:
	short					start_x, start_y;
	short					last_x, last_y;

	MGlobal::ListAdjustment	m_listAdjustment;
	M3dView					view;
	
	MDagPath m_activeMeshPath;
	MFnMesh fcollide;
	char goCollide;
	MDoubleArray curveLen;
	Matrix44F mat;
	MPoint _worldEye;
	double clipNear, clipFar;
	unsigned mOpt, m_numSeg;
	unsigned m_cullSelection, m_multiCreate;
	int m_groupCount;
	
	void resize();
	void grow();
	void flood();
	void snap();
	void erase();
	void move();
	void extractSelected();
	void erectSelected();
	void rotateAroundAxis(short axis);
	void moveAlongAxis(short axis);
	void startProcessSelect();
	void processSelect();
	char validateViz(const MSelectionList &sels);
	char validateCollide(const MSelectionList &sels);
	char validateSelection();
	void sendCullSurface();
	void smoothSelected();
	void selectGround();
	void startSelectGround();
	void setGrowOption(ProxyViz::GrowOption & opt);
	void finishGrow();
	
	ProxyViz *m_pViz;
	IntersectionGroup m_intersectionGrp;
	PseudoNoise pnoise;
	MPoint _lastHit;
	
	int _seed;
};

class proxyPaintContextCmd : public MPxContextCommand
{
public:	
						proxyPaintContextCmd();
	virtual MStatus		doEditFlags();
	virtual MStatus doQueryFlags();
	virtual MPxContext*	makeObj();
	static	void*		creator();
	virtual MStatus		appendSyntax();
	
protected:
    proxyPaintContext*		fContext;
};
