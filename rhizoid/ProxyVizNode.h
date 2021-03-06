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

#ifdef LINUX
#include <gl_heads.h>
#endif
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
#include "MForest.h"
#include "ExampVox.h"

namespace aphid {

class GrowOption;

class ProxyViz : public MPxLocatorNode, public MForest, public aphid::CachedExampParam
{
	M3dView _viewport;
	MMatrix _worldSpace, _worldInverseSpace;
	ExampVox * m_defExample;
	float m_transBuf[16];
	int m_iShowGrid;
	int m_iFastGround;
	bool m_toSetGrid;
	bool m_toCheckVisibility;
	bool m_hasParticle, m_enableCompute;
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
	virtual bool 	drawLast () const;
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
	static MObject outReplacePP;
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
	static MObject awmultiplier;
	static MObject aconvertPercentage;
	static MObject astandinNames;
    static MObject agroundMesh;
	static MObject agroundSpace;
	static MObject aplantTransformCache;
	static MObject aplantIdCache;
	static MObject aplantTriangleIdCache;
	static MObject aplantTriangleCoordCache;
	static MObject aplantOffsetCache;
	static MObject ainexamp;
    static MObject adisplayVox;
	static MObject acheckDepth;
	static MObject ainoverscan;
    static MObject aactivated;
    static MObject adrawDopSizeX;
	static MObject adrawDopSizeY;
	static MObject adrawDopSizeZ;
	static MObject adrawDopSize;
	static MObject aininstspace;
	static MObject adrawColorR;
	static MObject adrawColorG;
	static MObject adrawColorB;
	static MObject adrawColor;
	static MObject avoxactive;
	static MObject avoxvisible;
	static MObject avoxpriority;
	static MObject outValue1;
	static MObject outValue2;
	static MObject outValue;
	static	MTypeId		id;
	
	void adjustPosition(short x0, short y0, short x1, short y1, float clipNear, float clipFar);
	void rotateToDirection(short x0, short y0, short x1, short y1, float clipNear, float clipFar);
	void pressToSave();
	void pressToLoad();
	void beginPickInView();
	void processPickInView(const int & plantTyp);
	void endPickInView();
	
	const MMatrix & worldSpace() const;
	void setEnableCompute(bool x);
	
	void processDeselectSamples();
	void processReshuffle();
	void processFilterPortion(const float & x);
	void processFilterNoise(const GrowOption & param);
	void processFilterImage(const GrowOption & param);
	void processFlood(GrowOption & option);
	void processRemoveActivePlants();
	void processRemoveTypedPlants(const GrowOption & param);
	void processClearAllPlants();
	void processDeselectPlants();
	void processBrushRadius(const float & x);
	void processManipulatMode(ModifyForest::ManipulateMode x,
				GrowOption & option);
	void processViewDependentSelectSamples();
	void processSetShowGrid(int x);
	const int & getShowGrid() const;
	void processSetFastGround(int x);
	const int & getFastGround() const;
	void processBrushFalloff(float x);
	void processFilterPlantTypeMap(const std::vector<int> & indices,
				const std::vector<Vector3F> & colors);
	void processSampleColorChanges(const std::vector<Vector3F> & colors);
	bool processLoadExternal(const char * fileName);
	bool processStartEditOffset(const Ray & ray);
	
	bool isNullViz() const;
	
private:
    void drawBrush(M3dView & view);
	
private:
	std::string replaceEnvVar(const MString & filename) const;
    
	void updateWorldSpace(const MObject & thisNode);
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
	Ray getRayDisplace(Vector3F & v0, Vector3F & v1,
				short start_x, short start_y, short last_x, short last_y);
	
	void updateDrawSize(ExampVox * dst, const MObject & node);
	void updateDrawSize(ExampVox * dst, MDataBlock & block);
	void updateGeomBox(ExampVox * dst, const MObject & node);
	void updateGeomBox(ExampVox * dst, MDataBlock & block);
	void updateGeomDop(ExampVox * dst, const MObject & node);
	void updateGeomDop(ExampVox * dst, MDataBlock & block);
	/// extend, size and color if changed
	void updateDop(ExampVox * dst, const float * dopcorners);
	bool isDopParamChanged(ExampVox * dst, const float * dopcorners);
	
};

}
#endif        //  #ifndef PROXYVIZNODE_H
