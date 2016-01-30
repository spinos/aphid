/*
 *  PlantSelection.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Vector3F.h>
#include <WorldGrid.h>
#include <Array.h>
#include <SelectionContext.h>

namespace sdb {

/// (plant id, (transformation, plant type id, triangle bind id) )
typedef Triple<Matrix44F, int, int > PlantData;
class Plant : public Pair<int, PlantData>
{
public:
	
	const bool operator==(const Plant & another) const {
		return index == another.index;
	}
	
};

class PlantInstance
{
public:
	~PlantInstance()
	{
		delete m_backup;
	}
	
	PlantData * m_reference;
	PlantData * m_backup;
	float m_weight;
	
};

class PlantSelection {
	
	Vector3F m_center, m_direction;
	float m_radius;
	unsigned m_numSelected;
	WorldGrid<Array<int, Plant>, Plant > * m_grid;
	Array<int, PlantInstance> * m_plants;
	
public:
	PlantSelection(WorldGrid<Array<int, Plant>, Plant > * grid);
	virtual ~PlantSelection();
	
	void set(const Vector3F & center, const Vector3F & direction,
			const float & radius);
	void select(SelectionContext::SelectMode mode);
	void deselect();
	const unsigned & count() const;
	Array<int, PlantInstance> * data();
	void calculateWeight();
	
protected:

private:
	void select(const Coord3 & c, SelectionContext::SelectMode mode);
	void select(Plant * p);
};

}