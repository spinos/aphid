/*
 *  Forest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Forest.h"

namespace sdb {

Forest::Forest() 
{
	TreeNode::MaxNumKeysPerNode = 128;
    TreeNode::MinNumKeysPerNode = 16;

	m_grid = new WorldGrid<Array<int, Plant>, Plant >;
	m_ground = NULL;
}

Forest::~Forest() 
{
	delete m_grid;
	std::vector<Plant *>::iterator ita = m_plants.begin();
	for(;ita!=m_plants.end();++ita) delete *ita;
	std::vector<RotPosTri *>::iterator itb = m_pool.begin();
	for(;itb!=m_pool.end();++itb) {
		delete (*itb)->t1;
		delete (*itb)->t2;
		delete (*itb)->t3;
		delete (*itb);
	}
	if(m_ground) delete m_ground;
}

void Forest::resetGrid(float gridSize)
{
	m_grid->clear();
	m_grid->setGridSize(gridSize);
}

void Forest::finishGrid()
{
	m_grid->calculateBBox();
	std::cout<<"\n Forest grid bbox "<<m_grid->boundingBox();
}

void Forest::addPlant(const Quaternion & orientation, 
					const Vector3F & position,
					const int & triangleId)
{
	RotPosTri * d = new RotPosTri;
	*d->t1 = orientation;
	*d->t2 = position;
	*d->t3 = triangleId;
	m_pool.push_back(d);
	
	Plant * p = new Plant;
	p->key = m_plants.size();
	p->index = m_pool.back();
	m_plants.push_back(p);
	
	m_grid->insert((const float *)p->index->t2, p );
}

void Forest::resetGround()
{
	if(m_ground) delete m_ground;
	m_ground = new KdTree;
}

void Forest::addGroundMesh(ATriangleMesh * trimesh)
{ m_ground->addGeometry(trimesh); }

void Forest::finishGround()
{ m_ground->create(); }

const BoundingBox Forest::boundingBox() const
{ return m_grid->boundingBox(); }

unsigned Forest::numPlants() const
{ return m_plants.size(); }

}