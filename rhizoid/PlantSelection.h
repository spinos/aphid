/*
 *  PlantSelection.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <PlantCommon.h>
#include <SelectionContext.h>

namespace aphid {

class ForestCell;

class PlantSelection {

public:
	typedef sdb::Array<sdb::Coord2, PlantInstance> SelectionTyp;

private:	
	Vector3F m_center, m_direction;
	float m_radius, m_weightDecay;
	int m_numSelected;
    int m_typeFilter;
	sdb::WorldGrid<ForestCell, Plant > * m_grid;
	SelectionTyp * m_plants;
	
public:
	PlantSelection(sdb::WorldGrid<ForestCell, Plant > * grid);
	virtual ~PlantSelection();
	
    void setRadius(float x);
	void setCenter(const Vector3F & center, const Vector3F & direction);
	void select(SelectionContext::SelectMode mode);
	void selectByType(int x);
    void deselect();
	const int & numSelected() const;
	SelectionTyp * data();
	void calculateWeight();
	void select(Plant * p, const int & sd=0);
	const float & radius() const;
    
    void setTypeFilter(int x);
    
	bool touchCell(const Ray & incident, const sdb::Coord3 & c, 
					Vector3F & pnt);
	
protected:
	void updateNumSelected();
    
private:
	void selectInCell(const sdb::Coord3 & c, 
	            const SelectionContext::SelectMode & mode);
	void selectByTypeInCell(ForestCell * cell, int x);
	
};

}