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
#include <ViewDepthCull.h>
#include <DrawBox.h>
#include <DrawCircle.h>

class CircleCurve;

class DrawForest : public sdb::ModifyForest, public ViewDepthCull, public DrawBox, public DrawCircle
{
	
    Matrix44F m_useMat;
	float m_transbuf[16];
	float m_scalbuf[3];
	
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
	void drawBrush();
	void drawDepthCull(double * localTm);
	bool isVisibleInView(sdb::Plant * pl, 
					const float lowLod, const float highLod);
	
private:
    void drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components);
	void drawPlants(sdb::Array<int, sdb::Plant> * cell);
	void drawPlant(sdb::PlantData * data);
	void drawWiredPlants(sdb::Array<int, sdb::Plant> * cell);
	void drawWiredPlant(sdb::PlantData * data);
	void drawPlantBox(sdb::PlantData * data);
	void drawPlant(const ExampVox * v, sdb::PlantData * data);
	
};