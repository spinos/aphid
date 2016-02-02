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
class CircleCurve;

class DrawForest : public sdb::ModifyForest, public ViewDepthCull
{
	
    Matrix44F m_useMat;
    BoundingBox m_defBox;
	float m_boxExtent;
	float m_transbuf[16];
	float m_scalbuf[3];
	CircleCurve * m_circle;
	
public:
    DrawForest();
    virtual ~DrawForest();
    
protected:
	void setScaleMuliplier(float x, int idx);
    void drawGround();
	BoundingBox * defBoxP();
	const BoundingBox & defBox() const;
	void draw_solid_box() const;
	void draw_a_box() const;
	void draw_coordsys() const;
	int activePlantId() const;
	virtual float plantSize(int idx) const;
	Vector3F plantCenter(int idx) const;
	float plantExtent(int idx) const;
	void drawPlants();
	void drawWiredPlants();
	void drawGridBounding();
	void drawGrid();
	void drawActivePlants();
	void drawViewFrustum();
	void drawBrush();
	void drawDepthCull(double * localTm);
	bool isVisibleInView(sdb::Plant * pl);
    void calculateDefExtent();
	
private:
    void drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components);
	void drawPlants(sdb::Array<int, sdb::Plant> * cell);
	void drawPlant(sdb::PlantData * data);
	void drawWiredPlants(sdb::Array<int, sdb::Plant> * cell);
	void drawWiredPlant(sdb::PlantData * data);
	void drawBounding(const BoundingBox & b) const;
	void drawCircle() const;
    
};