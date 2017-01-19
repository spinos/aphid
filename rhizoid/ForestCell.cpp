/*
 *  ForestCell.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "ForestCell.h"
#include <PlantCommon.h>

namespace aphid {

ForestCell::ForestCell(Entity * parent) : sdb::Array<sdb::Coord2, Plant>(parent)
{}

ForestCell::~ForestCell()
{}

}
