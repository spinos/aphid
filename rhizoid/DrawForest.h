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
#include <DrawBox.h>
#include <DrawCircle.h>

namespace aphid {

class CircleCurve;

class DrawForest : public ModifyForest, public ViewObscureCull, public DrawBox, public DrawCircle
{
	
    Matrix44F m_useMat;
    float m_wireColor[3];
	float m_transbuf[16];
	float m_scalbuf[3];
    float m_showVoxLodThresold;
	
public:
    DrawForest();
    virtual ~DrawForest();
    
protected:
	void setScaleMuliplier(float x, float y, float z);
    void drawGround();
	float plantExtent(int idx) const;
	void drawPlants();
	void drawWiredPlants();
	void drawGridBounding();
	void drawGrid();
	void drawActivePlants();
	void drawViewFrustum();
	bool isVisibleInView(Plant * pl, 
					const float lowLod, const float highLod);
	void setShowVoxLodThresold(const float & x);
    void drawBrush();
	void setWireColor(const float & r, const float & g, const float & b);
    
private:
    void drawFace(const int & geoId, const int & triId);
	void drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components);
	void drawPlants(sdb::Array<int, Plant> * cell);
	void drawPlant(PlantData * data);
	void drawWiredPlants(sdb::Array<int, Plant> * cell);
	void drawWiredPlant(PlantData * data);
	void drawPlantBox(PlantData * data);
	void drawPlant(const ExampVox * v, PlantData * data);
	
};

}