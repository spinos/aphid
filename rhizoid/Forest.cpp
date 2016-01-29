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
    KdTree::MaxBuildLevel = 20;
	KdTree::NumPrimitivesInLeafThreashold = 15;
	
	m_grid = new WorldGrid<Array<int, Plant>, Plant >;
	m_ground = NULL;
}

Forest::~Forest() 
{
	delete m_grid;
    
	std::vector<Plant *>::iterator ita = m_plants.begin();
	for(;ita!=m_plants.end();++ita) delete *ita;
    m_plants.clear();
    
	std::vector<RotPosTri *>::iterator itb = m_pool.begin();
	for(;itb!=m_pool.end();++itb) {
		delete (*itb)->t1;
		delete (*itb)->t2;
		delete (*itb)->t3;
		delete (*itb);
	}
    m_pool.clear();
    
    clearGroundMeshes();
    
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

const BoundingBox Forest::boundingBox() const
{ return m_grid->boundingBox(); }

unsigned Forest::numPlants() const
{ return m_plants.size(); }

unsigned Forest::numGroundMeshes() const
{ return m_grounds.size(); }

void Forest::clearGroundMeshes()
{
    std::vector<ATriangleMesh *>::iterator itg = m_grounds.begin();
    for(;itg!=m_grounds.end();++itg) delete *itg;
    m_grounds.clear();
}

void Forest::setGroundMesh(ATriangleMesh * trimesh, unsigned idx)
{ 
    if(idx >= numGroundMeshes() ) m_grounds.push_back(trimesh); 
    else m_grounds[idx] = trimesh;
}

ATriangleMesh * Forest::getGroundMesh(unsigned idx) const
{
    if(idx >= numGroundMeshes() ) return NULL;
    return m_grounds[idx];
}

void Forest::buildGround()
{
    if(m_ground) delete m_ground;
	m_ground = new KdTree;

    std::vector<ATriangleMesh *>::const_iterator it = m_grounds.begin();
    for(;it!=m_grounds.end();++it) m_ground->addGeometry(*it);

    m_ground->create();
}

}