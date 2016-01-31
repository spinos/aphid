/*
 *  Forest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Forest.h"
#include <TriangleRaster.h>
#include <BarycentricCoordinate.h>

namespace sdb {

Forest::Forest() 
{
	TreeNode::MaxNumKeysPerNode = 128;
    TreeNode::MinNumKeysPerNode = 16;
    KdTree::MaxBuildLevel = 20;
	KdTree::NumPrimitivesInLeafThreashold = 16;
	
	m_grid = new WorldGrid<Array<int, Plant>, Plant >;
	m_ground = NULL;
	m_seed = rand() % 999999;
	m_numPlants = 0;
	m_activePlants = new PlantSelection(m_grid);
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

void Forest::resetGrid(float gridSize)
{
	m_grid->clear();
	m_grid->setGridSize(gridSize);
}

void Forest::updateGrid()
{
	m_grid->calculateBBox();
	std::cout<<"\n Forest grid bbox "<<m_grid->boundingBox();
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
	if(m_ground->isEmpty() ) return false;
	if(numPlants() < 1) return false;
	
	m_intersectCtx.reset(ray);
	m_ground->intersect(&m_intersectCtx);
	
	if(!m_intersectCtx.m_success) {
/// empty previous selection if hit nothing
		if(mode == SelectionContext::Replace)
			m_activePlants->deselect();
		return false;
	}
	
	m_activePlants->set(m_intersectCtx.m_hitP, m_intersectCtx.m_hitN, 4.f);
	m_activePlants->select(mode);
	
	return true;
}

bool Forest::selectGroundFaces(const Ray & ray, SelectionContext::SelectMode mode)
{
	if(m_ground->isEmpty() ) return false;
	
	m_intersectCtx.reset(ray);
	m_ground->intersect(&m_intersectCtx);
	
	if(!m_intersectCtx.m_success) {
/// empty previous selection if hit nothing
		if(mode == SelectionContext::Replace)
			m_selectCtx.deselect();
		return false;
	}
	
	m_selectCtx.setSelectMode(mode);
	m_selectCtx.reset(m_intersectCtx.m_hitP, 4.f);
	m_selectCtx.setDirection(m_intersectCtx.m_hitN);
	
	//std::cout<<"\n select P "<<m_selectCtx.center();
	//std::cout<<"\n select r "<<m_selectCtx.radius();
	
	m_ground->select(&m_selectCtx);
	return true;
}

unsigned Forest::numActiveGroundFaces()
{ return m_selectCtx.countComponents(); }

SelectionContext * Forest::activeGround()
{ return &m_selectCtx; }

void Forest::growOnGround(GrowOption & option)
{
	if(numActiveGroundFaces() < 1) return;
	std::map<Geometry *, Sequence<unsigned> * >::iterator it = m_selectCtx.geometryBegin();
	for(; it != m_selectCtx.geometryEnd(); ++it) {
		growOnFaces(it->first, it->second, geomertyId(it->first), option);
	}
}

void Forest::growOnFaces(Geometry * geo, Sequence<unsigned> * components, 
						int geoId,
						GrowOption & option)
{
	GroundBind bind;
	ATriangleMesh * mesh = static_cast<ATriangleMesh *>(geo);
	Vector3F * p = mesh->points();
	components->begin();
	while(!components->end()) {
		if(option.m_alongNormal)
			option.m_upDirection = mesh->triangleNormal(components->key() );
		
		unsigned * tri = mesh->triangleIndices(components->key() );
		TriangleRaster trir;
		BarycentricCoordinate bar;
		if(trir.create(p[tri[0]], p[tri[1]], p[tri[2]] ) ) {
			bar.create(p[tri[0]], p[tri[1]], p[tri[2]] );
			
			bind.setGeomComp(geoId, components->key() );
			growOnTriangle(&trir, &bar, bind, option);
		}
		components->next();
	}
}

void Forest::growOnTriangle(TriangleRaster * tri, 
							BarycentricCoordinate * bar,
							GroundBind & bind,
							GrowOption & option)
{
	float sampleSize = plantSize(option.m_plantId) * .7f;
	int grid_x, grid_y;
	tri->gridSize(sampleSize, grid_x, grid_y);
	int numSamples = grid_x * grid_y;
	
	Vector3F *samples = new Vector3F[numSamples];
	char *hits = new char[numSamples];
	tri->genSamples(sampleSize, grid_x, grid_y, samples, hits);
	float scale;
	Matrix44F tm;
	for(int s = 0; s < numSamples; s++) {
		if(!hits[s]) continue;	
	
		Vector3F & pos = samples[s];
			
		randomSpaceAt(pos, option, tm, scale);
		float delta = option.m_marginSize + sampleSize * 1.4f * scale;
		if(closeToOccupiedPosition(pos, delta)) continue;
		
		bar->project(pos);
		bar->compute();
		if(!bar->insideTriangle()) continue;
		
		bind.m_w0 = bar->getV(0);
		bind.m_w1 = bar->getV(1);
		bind.m_w2 = bar->getV(2);
		
		addPlant(tm, bind, option.m_plantId);
	}
	delete[] samples;
	delete[] hits;
}

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

float Forest::plantSize(int idx) const
{ return 1.f; }

void Forest::randomSpaceAt(const Vector3F & pos, const GrowOption & option,
							Matrix44F & space, float & scale)
{
	*space.m(3, 0) =  pos.x;
	*space.m(3, 1) =  pos.y;
	*space.m(3, 2) =  pos.z;
	
	Vector3F up = option.m_upDirection;
	
	Vector3F side(1.f, 0.f, 0.f);
	if(up.x > 0.9f || up.x < -0.9f)
		side = Vector3F::YAxis;
		
	Vector3F front = side.cross(up);
	front.normalize();
	
	side = up.cross(front);
	side.normalize();
	
	scale = (m_pnoise.rfloat(m_seed)) * (option.m_maxScale - option.m_minScale)
				+ option.m_minScale;
	m_seed++;
	
	Vector3F vside(side.x, side.y, side.z);
	Vector3F vaxis(up.x, up.y, up.z);
	float ang = (m_pnoise.rfloat(m_seed) - 0.5f) * option.m_rotateNoise * 2.f * 3.14f;
	m_seed++;
	
	vside.rotateAroundAxis(vaxis, ang);
	
	side = vside;
	front = side.cross(up);
	front.normalize();

	*space.m(0, 0) = side.x * scale;
	*space.m(0, 1) = side.y * scale;
	*space.m(0, 2) = side.z * scale;
	*space.m(1, 0) = up.x * scale;
	*space.m(1, 1) = up.y * scale;
	*space.m(1, 2) = up.z * scale;
	*space.m(2, 0) = front.x * scale;
	*space.m(2, 1) = front.y * scale;
	*space.m(2, 2) = front.z * scale;
}

WorldGrid<Array<int, Plant>, Plant > * Forest::grid()
{ return m_grid; }

const unsigned & Forest::numActivePlants() const
{ return m_activePlants->count(); }

Array<int, PlantInstance> * Forest::activePlants()
{ return m_activePlants->data(); }

void Forest::movePlant(const Ray & ray,
						const Vector3F & displaceNear, const Vector3F & displaceFar,
						const float & clipNear, const float & clipFar)
{
	if(numActivePlants() < 1 ) return;
	if(m_ground->isEmpty() ) return;
	
	m_intersectCtx.reset(ray);
	m_ground->intersect(&m_intersectCtx);
	
	if(!m_intersectCtx.m_success) return;
	
	const float depth = m_intersectCtx.m_hitP.distanceTo(ray.m_origin);
	const Vector3F disp = displaceNear 
					+ (displaceFar - displaceNear) * depth / (clipFar-clipNear);
	
	m_activePlants->set(m_intersectCtx.m_hitP, m_intersectCtx.m_hitN, 4.f);
	m_activePlants->calculateWeight();
	
	Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			Vector3F pos = plantd->t1->getTranslation()
						 + disp * wei;
						 
			m_closestPointTest.reset(pos, 1e8f);
			m_ground->closestToPoint(&m_closestPointTest);
			if(m_closestPointTest._hasResult) {
				plantd->t1->setTranslation(m_closestPointTest._hitPoint );
				PlantData * back = arr->value()->m_backup->index;
			
				Plant * moved = m_grid->displace(arr->value()->m_reference,
								m_closestPointTest._hitPoint, 
								back->t1->getTranslation() );
				if(moved) arr->value()->m_reference = moved;
			}
		}
		arr->next();
	}
}

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

void Forest::growAt(const Ray & ray, GrowOption & option)
{
	if(m_ground->isEmpty() ) return;
	
	m_intersectCtx.reset(ray);
	m_ground->intersect(&m_intersectCtx);
	
	if(!m_intersectCtx.m_success) return;
	
	ATriangleMesh * mesh = static_cast<ATriangleMesh *>(m_intersectCtx.m_geometry);
	unsigned component = m_intersectCtx.m_componentIdx;
	if(option.m_alongNormal)
		option.m_upDirection = mesh->triangleNormal(component );
		
	Vector3F * p = mesh->points();
	unsigned * tri = mesh->triangleIndices(component );
	TriangleRaster trir;
	BarycentricCoordinate bar;
	if(!trir.create(p[tri[0]], p[tri[1]], p[tri[2]] ) ) return;
	
	bar.create(p[tri[0]], p[tri[1]], p[tri[2]] );
	
	GroundBind bind;
	bind.setGeomComp(geomertyId(mesh), component );
	
	if(option.m_multiGrow) growOnTriangle(&trir, &bar, bind, option);
	else {
		Matrix44F tm;
		float scale;
		randomSpaceAt(m_intersectCtx.m_hitP, option, tm, scale);
		float delta = option.m_marginSize + plantSize(option.m_plantId) * scale;
		if(closeToOccupiedPosition(m_intersectCtx.m_hitP, delta)) return;
		
		bar.project(m_intersectCtx.m_hitP);
		bar.compute();
		
		bind.m_w0 = bar.getV(0);
		bind.m_w1 = bar.getV(1);
		bind.m_w2 = bar.getV(2);
		
		addPlant(tm, bind, option.m_plantId);
	}
}

}