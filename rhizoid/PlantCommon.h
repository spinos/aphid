/*
 *  PlantCommon.h
 *  proxyPaint
 *
 *  Created by jian zhang on 12/3/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PLANT_COMMON_H
#define APH_PLANT_COMMON_H
#include <math/Matrix44F.h>
#include <sdb/Array.h>
#include <sdb/WorldGrid.h>

namespace aphid {

struct GroundBind {
	float m_w0, m_w1, m_w2;
	int m_geomComp;
/// real_pos <- surface_pos + offset_vec
	Vector3F m_offset;
	
	void setGeomComp(int geom, int comp)
	{
		m_geomComp = ((geom<<22) | comp);
	}
	void getGeomComp(int & geom, int & comp) const
	{
		geom = m_geomComp>>22;
		comp = (m_geomComp << 10)>>10;
	}
};

/// ((plant id, plant type id), (transformation, triangle bind) )
typedef sdb::Couple<Matrix44F, GroundBind > PlantData;
class Plant : public sdb::Pair<sdb::Coord2, PlantData>
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
	
	Plant * m_reference;
	Plant * m_backup;
	float m_weight;
	int m_seed;
	
};

}
#endif