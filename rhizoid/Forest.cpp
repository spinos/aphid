/*
 *  Forest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 * 
 * http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 * qw= sqrt(1 + m00 + m11 + m22) /2
 * qx = (m21 - m12)/( 4 *qw)
 * qy = (m02 - m20)/( 4 *qw)
 * qz = (m10 - m01)/( 4 *qw)
 */

#include "Forest.h"
#include "ExampVox.h"

namespace aphid {

Forest::Forest() 
{    
	m_grid = new sdb::WorldGrid<sdb::Array<int, Plant>, Plant >;
	m_numPlants = 0;
	m_activePlants = new PlantSelection(m_grid);
	m_selectCtx = new SphereSelectionContext;
    
	ExampVox * defE = new ExampVox;
	addPlantExample(defE);
	
	m_ground = new KdNTree<cvx::Triangle, KdNode4 >();
	
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
    
	delete m_ground;
}

void Forest::setSelectionRadius(float x)
{ m_activePlants->setRadius(x); }

void Forest::resetGrid(float x)
{
	m_grid->clear();
	m_grid->setGridSize(x);
	std::cout<<"\n reset grid "<<x;
	std::cout.flush();
}

void Forest::updateGrid()
{
	m_grid->calculateBBox();
	m_march.initialize(m_grid->boundingBox(), m_grid->gridSize());
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
		sdb::Array<int, Plant> * cell = m_grid->value();
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
	m_triangles.clear();
	BoundingBox gridBox;
	
	KdEngine engine;
	engine.buildSource<cvx::Triangle, ATriangleMesh >(&m_triangles, 
													gridBox,
													m_grounds);

	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	
	engine.buildTree<cvx::Triangle, KdNode4, 4>(m_ground, &m_triangles, gridBox, &bf);
}

bool Forest::selectTypedPlants(int x)
{
	if(numPlants() < 1) return false;
	m_activePlants->selectByType(x);
	return true;
}

bool Forest::selectPlants(const Ray & ray, SelectionContext::SelectMode mode)
{
	if(numPlants() < 1) return false;
	
	if(!intersectGround(ray) ) {
		if(!intersectGrid(ray) ) 
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
		//if(mode == SelectionContext::Replace)
		//	m_selectCtx->deselect();
		return false;
	}
	
	std::cout<<"\n Forest::selectGroundFaces "<<m_intersectCtx.m_hitP;
	
	m_selectCtx->reset(m_intersectCtx.m_hitP, m_activePlants->radius(),
		mode);

	KdEngine engine;
	engine.select<cvx::Triangle, KdNode4>(m_ground, m_selectCtx);
	return true;
}

unsigned Forest::numActiveGroundFaces()
{ return m_selectCtx->numSelected(); }

SphereSelectionContext * Forest::activeGround()
{ return m_selectCtx; }

bool Forest::closeToOccupiedPosition(const Vector3F & pos, 
					const float & minDistance)
{
	sdb::Coord3 c0 = m_grid->gridCoord((const float *)&pos);
	sdb::Array<int, Plant> * cell = m_grid->findCell(c0);
	if(testNeighborsInCell(pos, minDistance, cell) ) return true;
	
	BoundingBox b = m_grid->coordToGridBBox(c0);
	
	sdb::Coord3 c1 = c0;
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
					sdb::Array<int, Plant> * cell)
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

sdb::WorldGrid<sdb::Array<int, Plant>, Plant > * Forest::grid()
{ return m_grid; }

const unsigned & Forest::numActivePlants() const
{ return m_activePlants->numSelected(); }

sdb::Array<int, PlantInstance> * Forest::activePlants()
{ return m_activePlants->data(); }

KdNTree<cvx::Triangle, KdNode4 > * Forest::ground()
{ return m_ground; }

const KdNTree<cvx::Triangle, KdNode4 > * Forest::ground() const
{ return m_ground; }

IntersectionContext * Forest::intersection()
{ return &m_intersectCtx; }

PlantSelection * Forest::selection()
{ return m_activePlants; }

void Forest::removeAllPlants()
{
	m_activePlants->deselect();
	m_grid->clear();
	m_numPlants = 0;
}

int Forest::getBindPoint(Vector3F & pos, GroundBind * bind)
{
	int geom, component;
	bind->getGeomComp(geom, component);
	if(geom < 0 || geom > 999) return -1;
	if(geom >= numGroundMeshes() ) return 0;
	
	ATriangleMesh * mesh = m_grounds[geom];
	if(component < 0 || component >= mesh->numTriangles() ) return 0;
	unsigned * tri = mesh->triangleIndices(component);
	Vector3F * pnt = mesh->points();
	pos = pnt[tri[0]] * bind->m_w0
			+ pnt[tri[1]] * bind->m_w1
			+ pnt[tri[2]] * bind->m_w2;
	return 1;
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

bool Forest::closestPointOnGround(Vector3F & dest,
								const Vector3F & origin,
								const float & maxDistance)
{
	m_closestPointTest.reset(origin, maxDistance);
	KdEngine engine;
	engine.closestToPoint<cvx::Triangle>(m_ground, &m_closestPointTest);
	if(m_closestPointTest._hasResult)
		dest = m_closestPointTest._hitPoint;
	else 
		dest = origin;
	return m_closestPointTest._hasResult;
}

bool Forest::bindToGround(GroundBind * bind, const Vector3F & origin, Vector3F & dest)
{
	if(closestPointOnGround(dest, origin, 1e7f) ) {
		bind->setGeomComp(m_closestPointTest._igeometry, 
								m_closestPointTest._icomponent );
		bind->m_w0 = m_closestPointTest._contributes[0];
		bind->m_w1 = m_closestPointTest._contributes[1];
		bind->m_w2 = m_closestPointTest._contributes[2];
		return true;
	}
	
	return false;
}

void Forest::bindToGround(PlantData * plantd, const Vector3F & origin, Vector3F & dest)
{ bindToGround(plantd->t2, origin, dest); }

bool Forest::intersectGround(const Ray & ray)
{
    if(!m_ground) return false;
	if(m_ground->isEmpty() ) return false;

	std::cout<<"\n Forest::intersectGround "<<ray.m_origin
		<<" "<<ray.m_dir<<" "<<ray.m_tmax;
		
	m_intersectCtx.reset(ray, m_grid->gridSize() * 0.001f );
	KdEngine engine;
	engine.intersect<cvx::Triangle, KdNode4>(m_ground, &m_intersectCtx );
	
	if(!m_intersectCtx.m_success) std::cout<<"\n Forest::intersectGround result is false";
	
	return m_intersectCtx.m_success;
}

bool Forest::intersectGrid(const Ray & incident)
{
	std::cout<<"\n Forest::intersectGrid";
	if(!m_march.begin(incident)) return false;
	sdb::Sequence<sdb::Coord3> added;
	BoundingBox touchedBox;
	Vector3F pnt;
	while(!m_march.end() ) {
		const std::deque<Vector3F> coords = m_march.touched(selectionRadius(), touchedBox);

		std::deque<Vector3F>::const_iterator it = coords.begin();
		for(; it != coords.end(); ++it) {
			const sdb::Coord3 c = m_grid->gridCoord((const float *)&(*it));
/// already tested
			if(added.find(c)) continue;
            
			added.insert(c);

			if(m_activePlants->touchCell(incident, c, pnt) ) {
				std::cout<<"\n Forest::intersectGrid hit cell"<<c;
				m_intersectCtx.m_hitP = pnt;
				m_intersectCtx.m_hitN = Vector3F::YAxis;
				return true;
			}
		}
		
		m_march.step();
	}
	return false;
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

int Forest::numPlantExamples() const
{ return m_examples.size(); }

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

std::string Forest::groundBuildLog() const
{ 
    if(!m_ground) return " error ground Kdtree not built"; 
    return "";//m_ground->buildLog();
}

const sdb::VectorArray<cvx::Triangle> & Forest::triangles() const
{ return m_triangles; }

const float & Forest::gridSize() const
{ return m_grid->gridSize(); }

}
