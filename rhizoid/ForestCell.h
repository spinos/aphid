/*
 *  ForestCell.h
 *  proxyPaint
 *
 *  plant and sample storage in cell
 *
 *  Created by jian zhang on 1/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_FOREST_CELL_H
#define APH_FOREST_CELL_H

#include <sdb/Array.h>

namespace aphid {

namespace sdb {

class LodSampleCache;

}

class Plant;

class ForestCell : public sdb::Array<sdb::Coord2, Plant> {

	sdb::LodSampleCache * m_lodsamp;
	
public:
	ForestCell(Entity * parent = NULL);
	virtual ~ForestCell();
	
	template<typename T>
	void buildSamples(T * ground);
	
protected:

private:
};

template<typename T>
void ForestCell::buildSamples(T * ground)
{

}

}
#endif
