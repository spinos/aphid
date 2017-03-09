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
#include <ttg/GlobalHeightField.h>
#include <ttg/GenericTetraGrid.h>
#include <ttg/AdaptiveBccGrid3.h>
#include <ttg/TetrahedronDistanceField.h>
#include <ttg/TetraGridTriangulation.h>
#include <geom/ATriangleMesh.h>

namespace aphid {

namespace ttg {

LandBlock::LandBlock(sdb::Entity * parent) : sdb::Entity(parent)
{
	BoundingBox bx(-1023.99f, -1023.99f, -1023.99f, 1023.99f, 1023.99f, 1023.99f);
	m_bccg = new AdaptiveBccGrid3;
	m_bccg->fillBox(bx, 1024.f);
	m_bccg->build();
	
	m_tetg = new TetGridTyp;
	
	ttg::TetraMeshBuilder teter;
    teter.buildMesh(m_tetg, m_bccg);
	
	m_field = new FieldTyp;
	m_mesher = new MesherTyp;
	
	m_mesher->setGridField(m_tetg, m_field);
	
	m_frontMesh = new ATriangleMesh;
}

LandBlock::~LandBlock()
{
	delete m_bccg;
	delete m_tetg;
	delete m_field;
	delete m_mesher;
	delete m_frontMesh;
}

void LandBlock::processHeightField(const GlobalHeightField * elevation)
{
	for(int i=0;i<m_field->numNodes();++i) {
		DistanceNode & d = m_field->nodes()[i];
		d.val = elevation->sample(d.pos);
	}
	
}

void LandBlock::triangulate()
{
	m_mesher->triangulate();
	m_mesher->dumpFrontTriangleMesh(m_frontMesh);
	m_frontMesh->calculateVertexNormals();
}

const LandBlock::TetGridTyp * LandBlock::grid() const
{ return m_tetg; }

const LandBlock::FieldTyp * LandBlock::field() const
{ return m_field; }

const ATriangleMesh * LandBlock::frontMesh() const
{ return m_frontMesh; }

}

}
