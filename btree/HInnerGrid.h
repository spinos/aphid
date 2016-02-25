/*
 *  HInnerGrid.h
 *  julia
 *
 *  Created by jian zhang on 1/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  child of world grid
 */

#pragma once

#include <HOocArray.h>
#include "Entity.h"

namespace aphid {
namespace sdb {

template <int DataRank, int NRows, int BufSize>
class HInnerGrid : public HOocArray<DataRank, NRows, BufSize>, public Entity {

public:
	HInnerGrid(const std::string & name, Entity * parent);
	virtual ~HInnerGrid();
	
protected:

private:

};

template <int DataRank, int NRows, int BufSize>
HInnerGrid<DataRank, NRows, BufSize>::HInnerGrid(const std::string & name, Entity * parent) :
HOocArray<DataRank, NRows, BufSize>(name), Entity(parent)
{}

template <int DataRank, int NRows, int BufSize>
HInnerGrid<DataRank, NRows, BufSize>::~HInnerGrid()
{}

}
}
