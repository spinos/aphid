/*
 *  DrawForest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "Forest.h"

class DrawForest : public sdb::Forest {
	
	BoundingBox m_defBox;
	
public:
    DrawForest();
    virtual ~DrawForest();
    
protected:
    void drawGround();
	BoundingBox * defBoxP();
	const BoundingBox & defBox() const;
	void draw_solid_box() const;
	void draw_a_box() const;
	void draw_coordsys() const;
	int activePlantId() const;
	virtual float plantSize(int idx) const;
	void drawPlants();
	void drawWiredPlants();
	void drawGridBounding();
	void drawGrid();
	
private:
    void drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components);
	void drawPlants(sdb::Array<int, sdb::Plant> * cell);
	void drawPlant(sdb::PlantData * data);
	void drawWiredPlants(sdb::Array<int, sdb::Plant> * cell);
	void drawWiredPlant(sdb::PlantData * data);
	void drawBounding(const BoundingBox & b) const;
	
};