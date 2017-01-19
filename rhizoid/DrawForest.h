/*
 *  DrawForest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "ModifyForest.h"
#include <ViewObscureCull.h>
#include <ogl/DrawBox.h>
#include <ogl/DrawInstance.h>
#include <ogl/DrawCircle.h>

namespace aphid {

class ExampVox;
class ForestCell;
class CircleCurve;

class DrawForest : public ModifyForest, public ViewObscureCull, 
public DrawBox, public DrawCircle, public DrawInstance
{
	
    Matrix44F m_useMat;
    float m_wireColor[3];
	float m_transbuf[16];
    float m_showVoxLodThresold;
    bool m_enabled;
	
public:
    DrawForest();
    virtual ~DrawForest();
    
protected:
	void setScaleMuliplier(float x, float y, float z);
    void drawGround();
	float plantExtent(int idx) const;
	void drawSolidPlants();
	void drawWiredPlants();
	void drawGridBounding();
	void drawGrid();
	void drawActivePlants();
	void drawViewFrustum();
	bool isVisibleInView(Plant * pl, 
					const ExampVox * v,
					const float lowLod, const float highLod);
	void setShowVoxLodThresold(const float & x);
    void drawBrush();
	void setWireColor(const float & r, const float & g, const float & b);
    void enableDrawing();
    void disableDrawing();
    
private:
    void drawFace(const int & geoId, const int & triId);
	void drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components);
	void drawPlantsInCell(ForestCell * cell,
					const BoundingBox & box);
	void drawPlant(PlantData * data,
					const ExampVox * v);
	void drawWiredPlants(ForestCell * cell);
	void drawWiredPlant(PlantData * data,
					const ExampVox * v);
	void drawPlantBox(PlantData * data,
					const ExampVox * v);
	void drawLODPlant(PlantData * data,
					const ExampVox * v);
	void drawPlantSolidBoundInCell(ForestCell * cell);
	void drawPlantSolidBound(PlantData * data,
					const ExampVox * v);

};

}