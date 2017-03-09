/*
 *  LandBlock.cpp
 *  
 *  a single piece of land
 *
 *  Created by jian zhang on 3/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
 
#include "LandBlock.h"
#include <ttg/TetraMeshBuilder.h>

namespace aphid {

namespace ttg {

LandBlock::LandBlock(sdb::Entity * parent) : sdb::Entity(parent)
{
	BoundingBox bx(-1023.99f, -1023.99f, -1023.99f, 1023.99f, 1023.99f, 1023.99f);
	m_bccg.fillBox(bx, 1024.f);
	m_bccg.build();
	
	ttg::TetraMeshBuilder teter;
    teter.buildMesh(&m_tetg, &m_bccg);
}

LandBlock::~LandBlock()
{}

const ttg::GenericTetraGrid<float > * LandBlock::grid() const
{ return &m_tetg; }

}

}
