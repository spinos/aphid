/*
 *  proxyPaintTool.h
 *
 *  Created by jian zhang on 6/3/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef RHI_BOX_PAINT_TOOL_H
#define RHI_BOX_PAINT_TOOL_H

#ifdef LINUX
#include <gl_heads.h>
#endif
#include <maya/MPxContext.h>
#include <maya/M3dView.h>
#include <maya/MGlobal.h>
#include <maya/MSceneMessage.h>
#include "GrowOption.h"
#include "ExampleWorks.h"

namespace aphid {
 
class Ray;

}

class proxyPaintContext : public MPxContext, public ExampleWorks
{
    
	M3dView					view;
	
	double clipNear, clipFar;
	
    enum Operation {
        opUnknown = 0,
        opErase = 1,
        opSelect = 2,
        opResize = 3,
        opMove = 4,
        opRotateY = 5,
        opRotateZ = 6,
        opRotateX = 7,
        opResizeBrush = 8,
        opSelectGround = 9,
        opReplace = 10,
        opCreate = 11,
        opSelectByType= 12,
		opRandResize = 13,
		opRandMove = 14,
		opRandRotate = 15,
		opErect = 16,
		opRaise = 17,
		opDepress = 18,
		opBundleResize = 19,
		opBundleRotate = 20,
        opBundleTranslate = 21,
		opInjectParticle = 97,
		opInjectTransform = 98,
        opClean = 99,
        opFlood = 100,
        opExtract = 102,
        opDiscardSampleSelection = 103,
        opDiscardPlantSelection = 104,
        opRotateToDir = 105,
		opCleanByType = 106,
		opClearOffset = 107,
		opReshuffle = 108
	};
    
    Operation m_currentOpt, mOpt;
    
    aphid::GrowOption m_growOpt;
    
	int m_extractGroupCount;
	short					start_x, start_y;
	short					last_x, last_y;

	MGlobal::ListAdjustment	m_listAdjustment;
	MCallbackId fBeforeNewCB;
	MCallbackId fBeforeOpenCB;
	
public:
					proxyPaintContext();
	virtual ~proxyPaintContext();
	virtual void	toolOnSetup( MEvent & event );
	virtual MStatus	doPress( MEvent & event );
	virtual MStatus	doDrag( MEvent & event );
	virtual MStatus	doRelease( MEvent & event );
	virtual MStatus	doEnterRegion( MEvent & event );
	
	virtual	void	getClassName(MString & name) const;
	
	void setOperation(short val);
	unsigned getOperation() const;
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
	void setMultiCreate(unsigned val);
	unsigned getMultiCreate() const;
	void setInstanceGroupCount(unsigned val);
    unsigned getInstanceGroupCount() const;
	void setWriteCache(MString filename);
	void setReadCache(MString filename);
	void clearAllPlants();
	void setMinCreateMargin(float x);
	const float & minCreateMargin();
    void setMaxCreateMargin(float x);
	const float & maxCreateMargin();
	void setPlantType(int x);
	const int & plantType() const;
	void setStickToGround(bool x);
	const bool & stickToGround() const;
	void selectViz();
	const float & noiseFrequency() const;
	const float & noiseLacunarity() const;
	const int & noiseOctave() const;
	const float & noiseLevel() const;
	const float & noiseGain() const;
	const float & noiseOriginX() const;
	const float & noiseOriginY() const;
	const float & noiseOriginZ() const;
	void setNoiseFrequency(float x);
	void setNoiseLacunarity(float x);
	void setNoiseOctave(int x);
	void setNoiseLevel(float x);
	void setNoiseGain(float x);
	void setNoiseOriginX(float x);
	void setNoiseOriginY(float x);
	void setNoiseOriginZ(float x);
	void setNoiseOrigin(float x, float y, float z);
	void setImageSamplerName(MString filename);
	MString imageSamplerName() const;
	void reshuffleSamples();
	void viewDependentSelectSamples();
	void setFilterPortion(float x);
	float filterPortion() const;
	int numVisibleSamples();
	void setShowVizGrid(int x);
	int getShowVizGrid();
	void setEditVizGround(int x);
	int getEditVizGround();
	void setBrushFalloff(float x);
	float getBrushFalloff();
	void setShowVoxelThreshold(float x);
	float zenithNoise() const;
	void setZenithNoise(float x);
	
private:
	void resize(bool isBundled);
	void grow();
	void flood();
	void snap();
	void erase();
	void move();
	void extractSelected();
	void rotateByStroke();
	void erect();
	void rotateAroundAxis(short axis);
	void moveAlongAxis(short axis);
	void startProcessSelect();
	void processSelect();
    void processSelectByType();
	void smoothSelected();
	void processSelectGround();
	void startSelectGround();
	void replace();
	void startResizeBrush();
    void processResizeBrush();
    void discardSampleSelection();
    void discardPlantSelection();
	void injectSelectedParticle();
    void injectSelectedTransform();
	void resizeSelectedRandomly();
	void moveRandomly();
	void rotateRandomly();
	void clearOffset();
	void raiseOffset();
	void depressOffset();
	bool rejectSmallDragDistance(int d = 2) const;
	void clearByType();
	void clearBySelections();
	void setManipulator(int x);
	void startRotate();
	void processRotate();
    void startTranslate();
	void processTranslate();
	void startResize();
	void processResize();
	void processFilterNoise();
    aphid::Ray getIncidentAt(int x, int y);
	void startEditOffset();
    
	void attachSceneCallbacks();
	void detachSceneCallbacks();
	static void releaseCallback(void* clientData);
	
};
#endif        //  #ifndef RHI_BOX_PAINT_TOOL_H

