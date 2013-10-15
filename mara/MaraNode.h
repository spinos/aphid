#ifndef MallardVIZNODE_H
#define MallardVIZNODE_H

/*
 *  MallardVizNode.h
 *  MallardPaint
 *
 *  Created by jian zhang on 3/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MMatrixArray.h>
#include <maya/MIntArray.h>
#include <maya/MFloatArray.h>
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include "AllMath.h"

class MlDrawer;
class MlScene;
class MallardViz : public MPxLocatorNode
{
public:
	MallardViz();
	virtual ~MallardViz(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 

	static  void *          creator();
	static  MStatus         initialize();
	
	static MObject acachename;
	static MObject aframe;
	static MObject ainmesh;
	static MObject outValue;
	
	void setCullMesh(MDagPath mesh);
	
public: 
	static	MTypeId		id;
	
private:
	char hasDisplayMesh() const;
	
	MMatrix _worldSpace, _worldInverseSpace;
	M3dView _viewport;
	char fHasView;
	
	void loadCache(const char* filename);
	void loadScene(const char* filename);
	void updateWorldSpace();
	MMatrix localizeSpace(const MMatrix & s) const;
	MMatrix worldizeSpace(const MMatrix & s) const;
	void useActiveView();

	MObject m_bodyMesh;
	MlDrawer * m_cache;
	MlScene * m_scene;
};
#endif        //  #ifndef MallardVIZNODE_H
