#ifndef PROXYVIZNODE_H
#define PROXYVIZNODE_H

/*
 *  proxyVizNode.h
 *  proxyPaint
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
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MSceneMessage.h>
#include <Matrix44F.h>
#include <Vector3F.h>
#include <depthCut.h>
#include "MForest.h"

class MItMeshPolygon;

class ProxyViz : public MPxLocatorNode, public MForest
{
	M3dView _viewport;
	MMatrix _worldSpace, _worldInverseSpace;
	double m_materializePercentage;
	bool m_toSetGrid;
	bool m_hasCamera;
	
public:
	ProxyViz();
	virtual ~ProxyViz(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 
	virtual MStatus connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc );
	virtual MStatus connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc );
	
	static  void *          creator();
	static  MStatus         initialize();

	static  MObject         abboxminx;
	static  MObject         abboxminy;
	static  MObject         abboxminz;
	static  MObject         abboxmaxx;
	static  MObject         abboxmaxy;
	static  MObject         abboxmaxz;
	static MObject outPositionPP;
	static MObject outScalePP;
	static MObject outRotationPP;
	static MObject acachename;
	static MObject adumpname;
	static MObject acameraspace;
	static MObject alodgatehigh;
	static MObject alodgatelow;
	static MObject ahapeture;
	static MObject avapeture;
	static MObject afocallength;
	static MObject axmultiplier;
	static MObject aymultiplier;
	static MObject azmultiplier;
	static MObject agroupcount;
	static MObject astarttime;
	static MObject ainstanceId;
	static MObject aenablecull;
	static MObject ainmesh;
	static MObject aconvertPercentage;
	static MObject astandinNames;
    static MObject agroundMesh;
	static MObject aplantTransformCache;
	static MObject aplantIdCache;
	static MObject aplantTriangleIdCache;
	static MObject aplantTriangleCoordCache;
	static MObject outValue;
	static	MTypeId		id;
	
	char isBoxInView(const MPoint &pos, float threshold, short xmin, short ymin, short xmax, short ymax);
	void adjustPosition(short x0, short y0, short x1, short y1, float clipNear, float clipFar, Matrix44F &mat);
	void pressToSave();
	void pressToLoad();
	void setCullMesh(MDagPath mesh);
	
	const MMatrix & worldSpace() const;
	
private:
	char hasDisplayMesh() const;
	std::string replaceEnvVar(const MString & filename) const;
    
	MMatrixArray _spaces;
	MFloatArray _details;
	MIntArray _randNums;
	
	char *fVisibleTag;
	char _firstLoad, fHasView;
	
	void calculateLOD(const MMatrix & cameraInv, const float & h_fov, const float & aspectRatio, const float & detail, const int & enableViewFrustumCulling);
	void updateWorldSpace();
	MMatrix localizeSpace(const MMatrix & s) const;
	MMatrix worldizeSpace(const MMatrix & s) const;
	void useActiveView();
	void vizViewFrustum(MObject & node);
	DepthCut * fCuller;
	MObject fDisplayMesh;
	
	void attachSceneCallbacks();
	void detachSceneCallbacks();
	static void releaseCallback(void* clientData);

	MCallbackId fBeforeSaveCB;
	void saveInternal();
	bool loadInternal(MDataBlock& block);
	
};
#endif        //  #ifndef PROXYVIZNODE_H
