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
#include "MForest.h"

class MItMeshPolygon;

class ProxyViz : public MPxLocatorNode, public MForest
{
	M3dView _viewport;
	MMatrix _worldSpace, _worldInverseSpace;
	float m_transBuf[16];
	bool m_toSetGrid;
	bool m_toCheckVisibility;
	bool m_hasCamera, m_hasParticle, m_enableCompute;
	char _firstLoad, fHasView;
	
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
	static MObject aradiusMult;
	static MObject outPositionPP;
	static MObject outScalePP;
	static MObject outRotationPP;
	static MObject acachename;
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
	static MObject ainstanceId;
	static MObject ainmesh;
	static MObject aconvertPercentage;
	static MObject astandinNames;
    static MObject agroundMesh;
	static MObject agroundSpace;
	static MObject aplantTransformCache;
	static MObject aplantIdCache;
	static MObject aplantTriangleIdCache;
	static MObject aplantTriangleCoordCache;
	static MObject ainexamp;
    static MObject adisplayVox;
	static MObject outValue1;
	static MObject outValue;
	static	MTypeId		id;
	
	void adjustPosition(short x0, short y0, short x1, short y1, float clipNear, float clipFar);
	void pressToSave();
	void pressToLoad();
	void beginPickInView();
	void processPickInView(const int & plantTyp);
	void endPickInView();
	
	const MMatrix & worldSpace() const;
	void setEnableCompute(bool x);
	
private:
	char hasDisplayMesh() const;
	std::string replaceEnvVar(const MString & filename) const;
    
	void updateWorldSpace();
	MMatrix localizeSpace(const MMatrix & s) const;
	MMatrix worldizeSpace(const MMatrix & s) const;
	void useActiveView();
	void updateViewFrustum(MObject & node);
	void updateViewFrustum(const MDagPath & cameraPath);
	
	MObject fDisplayMesh;
	
	void attachSceneCallbacks();
	void detachSceneCallbacks();
	static void releaseCallback(void* clientData);

	MCallbackId fBeforeSaveCB;
	void saveInternal();
	bool loadInternal(MDataBlock& block);
	void updateGeomBox(ExampVox * dst, MObject & node);
	
};
#endif        //  #ifndef PROXYVIZNODE_H
