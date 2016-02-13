/*
 *  Forest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Forest.h"
#include "ExampVox.h"

namespace sdb {

Forest::Forest() 
{
	TreeNode::MaxNumKeysPerNode = 128;
    TreeNode::MinNumKeysPerNode = 16;
    KdTree::MaxBuildLevel = 25;
	KdTree::NumPrimitivesInLeafThreashold = 16;
	
	m_grid = new WorldGrid<Array<int, Plant>, Plant >;
	m_ground = NULL;
	m_numPlants = 0;
	m_activePlants = new PlantSelection(m_grid);
    m_selectCtx.setRadius(8.f);
	
	ExampVox * defE = new ExampVox;
	addPlantExample(defE);
}

Forest::~Forest() 
{
	delete m_grid;
	delete m_activePlants;
    
	std::vector<Plant *>::iterator ita = m_plants.begin();
	for(;ita!=m_plants.end();++ita) delete *ita;
    m_plants.clear();
    
	std::vector<PlantData *>::iterator itb = m_pool.begin();
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

void Forest::setSelectionRadius(float x)
{ 
    m_selectCtx.setRadius(x); 
    m_activePlants->setRadius(x);
}

void Forest::resetGrid(float gridSize)
{
	m_grid->clear();
	m_grid->setGridSize(gridSize);
}

void Forest::updateGrid()
{
	m_grid->calculateBBox();
	// std::cout<<"\n Forest grid bbox "<<m_grid->boundingBox();
}

const BoundingBox & Forest::gridBoundingBox() const
{ return m_grid->boundingBox(); }

unsigned Forest::numPlants() const
{ return m_numPlants; }

void Forest::updateNumPlants()
{
	m_numPlants = 0;
	m_grid->begin();
	while(!m_grid->end() ) {
		Array<int, Plant> * cell = m_grid->value();
		m_numPlants += cell->size();
		m_grid->next();
	}
}

unsigned Forest::numCells()
{ return m_grid->size(); }

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

const std::vector<ATriangleMesh *> & Forest::groundMeshes() const
{ return m_grounds; }

void Forest::buildGround()
{
    if(m_ground) delete m_ground;
	m_ground = new KdTree;

    std::vector<ATriangleMesh *>::const_iterator it = m_grounds.begin();
    for(;it!=m_grounds.end();++it) m_ground->addGeometry(*it);

    m_ground->create();
}

bool Forest::selectPlants(const Ray & ray, SelectionContext::SelectMode mode)
{
	if(numPlants() < 1) return false;
	
	if(!intersectGround(ray) ) {
/// empty previous selection if hit nothing
		if(mode == SelectionContext::Replace)
			m_activePlants->deselect();
		return false;
	}
	
	m_activePlants->setCenter(m_intersectCtx.m_hitP, m_intersectCtx.m_hitN);
	m_activePlants->select(mode);
	
	return true;
}

bool Forest::selectGroundFaces(const Ray & ray, SelectionContext::SelectMode mode)
{
	if(!intersectGround(ray) ) {
/// empty previous selection if hit nothing
		if(mode == SelectionContext::Replace)
			m_selectCtx.deselect();
		return false;
	}
	
	m_selectCtx.setSelectMode(mode);
	m_selectCtx.setCenter(m_intersectCtx.m_hitP);
	m_selectCtx.setDirection(m_intersectCtx.m_hitN);

	m_ground->select(&m_selectCtx);
	return true;
}

unsigned Forest::numActiveGroundFaces()
{ return m_selectCtx.countComponents(); }

SelectionContext * Forest::activeGround()
{ return &m_selectCtx; }

bool Forest::closeToOccupiedPosition(const Vector3F & pos, 
					const float & minDistance)
{
	Coord3 c0 = m_grid->gridCoord((const float *)&pos);
	Array<int, Plant> * cell = m_grid->findCell(c0);
	if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	
	BoundingBox b = m_grid->coordToGridBBox(c0);
	
	Coord3 c1 = c0;
	if(pos.x - minDistance < b.getMin(0) ) {
		 c1.x = c0.x - 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	}
	if(pos.x + minDistance > b.getMax(0) ) {
		 c1.x = c0.x + 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	}
	c1.x = c0.x;
	if(pos.y - minDistance < b.getMin(1) ) {
		 c1.y = c0.y - 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	}
	if(pos.y + minDistance > b.getMax(1) ) {
		 c1.y = c0.y + 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	}
	c1.y = c0.y;
	if(pos.z - minDistance < b.getMin(2) ) {
		 c1.z = c0.z - 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	}
	if(pos.z + minDistance > b.getMax(2) ) {
		 c1.z = c0.z + 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	}
	return false;
}

bool Forest::testNeighborsInCell(const Vector3F & pos, 
					const float & minDistance,
					Array<int, Plant> * cell)
{
	if(!cell) return false;
	if(cell->isEmpty() ) return false;
	cell->begin();
	while(!cell->end()) {
		PlantData * d = cell->value()->index;
		float scale = d->t1->getSide().length();
		if(pos.distanceTo(d->t1->getTranslation() ) - plantSize(*d->t3) * scale < minDistance) return true;
		  
		cell->next();
	}
	return false;
}

const float & Forest::plantSize(int idx) const
{ return m_examples[idx]->geomSize(); }

WorldGrid<Array<int, Plant>, Plant > * Forest::grid()
{ return m_grid; }

const unsigned & Forest::numActivePlants() const
{ return m_activePlants->numSelected(); }

Array<int, PlantInstance> * Forest::activePlants()
{ return m_activePlants->data(); }

KdTree * Forest::ground()
{ return m_ground; }

IntersectionContext * Forest::intersection()
{ return &m_intersectCtx; }

PlantSelection * Forest::selection()
{ return m_activePlants; }

int Forest::geomertyId(Geometry * geo) const
{
	int i = 0;
	std::vector<ATriangleMesh *>::const_iterator it = m_grounds.begin();
	for(;it!=m_grounds.end(); ++it) {
		if(*it == geo) return i;
		i++;
	}
	return -1;
}

void Forest::removeAllPlants()
{
	m_activePlants->deselect();
	m_grid->clear();
	m_numPlants = 0;
}

bool Forest::getBindPoint(Vector3F & pos, GroundBind * bind)
{
	int geom, component;
	bind->getGeomComp(geom, component);
	if(geom < 0 || geom >= numGroundMeshes() ) return false;
	
	ATriangleMesh * mesh = m_grounds[geom];
	if(component < 0 || component >= mesh->numTriangles() ) return false;
	unsigned * tri = mesh->triangleIndices(component);
	Vector3F * pnt = mesh->points();
	pos = pnt[tri[0]] * bind->m_w0
			+ pnt[tri[1]] * bind->m_w1
			+ pnt[tri[2]] * bind->m_w2;
	return true;
}

void Forest::displacePlantInGrid(PlantInstance * inst )
{
	PlantData * back = inst->m_backup->index;
	PlantData * cur = inst->m_reference->index;
	Plant * moved = m_grid->displace(inst->m_reference,
					cur->t1->getTranslation(), 
					back->t1->getTranslation() );
	if(moved) inst->m_reference = moved;
}

void Forest::bindToGround(PlantData * plantd, const Vector3F & origin, Vector3F & dest)
{
	m_closestPointTest.reset(origin, 1e8f);
	m_ground->closestToPoint(&m_closestPointTest);
	if(m_closestPointTest._hasResult) {
	
		GroundBind * bind = plantd->t2;
		bind->setGeomComp(geomertyId(m_closestPointTest._geom), 
								m_closestPointTest._icomponent );
		bind->m_w0 = m_closestPointTest._contributes[0];
		bind->m_w1 = m_closestPointTest._contributes[1];
		bind->m_w2 = m_closestPointTest._contributes[2];
		
		dest = m_closestPointTest._hitPoint;
	}
	else 
		dest = origin;
}

bool Forest::intersectGround(const Ray & ray)
{
    if(!m_ground) return false;
	if(m_ground->isEmpty() ) return false;
	
	m_intersectCtx.reset(ray);
	m_ground->intersect(&m_intersectCtx );
	
	return m_intersectCtx.m_success;
}

void Forest::addPlant(const Matrix44F & tm,
					const GroundBind & bind,
					const int & plantTypeId)
{
	PlantData * d = new PlantData;
	*d->t1 = tm;
	*d->t2 = bind;
	*d->t3 = plantTypeId;
	m_pool.push_back(d);
	
	Plant * p = new Plant;
	p->key = m_plants.size();
	p->index = m_pool.back();
	m_plants.push_back(p);
	
	const Vector3F at = tm.getTranslation();
	m_grid->insert((const float *)&at, p );
    m_activePlants->select(p);
}

const float & Forest::selectionRadius() const
{ return m_activePlants->radius(); }

const Vector3F & Forest::selectionCenter() const
{ return m_intersectCtx.m_hitP; }

const Vector3F & Forest::selectionNormal() const
{ return m_intersectCtx.m_hitN; }

bool Forest::isGroundEmpty() const
{
    if(!m_ground) return true;
    return m_ground->isEmpty();
}

void Forest::addPlantExample(ExampVox * x)
{
	if(m_exampleIndices.find(x) != m_exampleIndices.end() ) return;
	m_exampleIndices[x] = m_examples.size();
	m_examples.push_back(x);
}

ExampVox * Forest::plantExample(unsigned idx)
{ return m_examples[idx]; }

const ExampVox * Forest::plantExample(unsigned idx) const
{ return m_examples[idx]; }

void Forest::setSelectTypeFilter(int flt)
{ m_activePlants->setTypeFilter(flt); }

}