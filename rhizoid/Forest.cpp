/*
 *  Forest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/29/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Forest.h"
#include "ForestCell.h"
#include "ForestGrid.h"
#include "ExampVox.h"
#include <geom/ATriangleMesh.h>
#include <kd/ClosestToPointEngine.h>
#include <kd/IntersectEngine.h>
#include "SampleFilter.h"
#include <geom/AFrustum.h>

namespace aphid {

Forest::Forest() 
{    
	m_grid = new ForestGrid;
	m_numPlants = 0;
	m_sampleFlt = new SampleFilter;
	m_activePlants = new PlantSelection(m_grid, m_sampleFlt->plantTypeMap() );
	m_ground = new KdNTree<cvx::Triangle, KdNode4 >();
	m_lastPlantInd = -1;
}

Forest::~Forest() 
{
    deselectSamples();
    clearAllPlants();
    clearGroundMeshes();
    
    m_grid->clear();
    
	delete m_ground;
	delete m_activePlants;
	delete m_sampleFlt;
	delete m_grid;
    
}

void Forest::setSelectionRadius(float x)
{ m_activePlants->setRadius(x); }

void Forest::setSelectionFalloff(float x)
{ m_activePlants->setFalloff(x); }

void Forest::resetGrid(float x)
{
	m_grid->clear();
	m_grid->setGridSize(x);
	m_sampleFlt->computeGridLevelSize(gridSize(), plantSize(0) * 1.41f );
	std::cout<<"\n reset grid "<<gridSize()
			<<"\n sample level "<<sampleLevel();
	std::cout.flush();
}

void Forest::updateGrid()
{
	m_grid->calculateBBox();
	m_march.initialize(m_grid->boundingBox(), m_grid->gridSize());
}

const BoundingBox & Forest::gridBoundingBox() const
{ return m_grid->boundingBox(); }

const int & Forest::numPlants() const
{ return m_numPlants; }

void Forest::countNumPlants()
{ m_numPlants = m_grid->countPlants(); }

unsigned Forest::numCells()
{ return m_grid->size(); }

unsigned Forest::numGroundMeshes() const
{ return m_groundMeshes.size(); }

void Forest::clearGroundMeshes()
{
    std::vector<ATriangleMesh *>::iterator itg = m_groundMeshes.begin();
    for(;itg!=m_groundMeshes.end();++itg) {
		delete *itg;
	}
    m_groundMeshes.clear();
}

void Forest::setGroundMesh(ATriangleMesh * trimesh, unsigned idx)
{ 
    if(idx >= numGroundMeshes() ) m_groundMeshes.push_back(trimesh); 
    else m_groundMeshes[idx] = trimesh;
}

ATriangleMesh * Forest::getGroundMesh(const int & idx) const
{
    if(idx >= numGroundMeshes() ) {
		std::cout<<"\n Forest out-of-range geom "<<idx;
		return NULL;
	}
    return m_groundMeshes[idx];
}

const std::vector<ATriangleMesh *> & Forest::groundMeshes() const
{ return m_groundMeshes; }

void Forest::buildGround()
{
	m_triangles.clear();
	BoundingBox gridBox;
	
	KdEngine engine;
	engine.buildSource<cvx::Triangle, ATriangleMesh >(&m_triangles, 
													gridBox,
													m_groundMeshes);

	TreeProperty::BuildProfile bf;
	bf._maxLeafPrims = 64;
	
	engine.buildTree<cvx::Triangle, KdNode4, 4>(m_ground, &m_triangles, gridBox, &bf);
}

bool Forest::selectTypedPlants()
{
	if(numPlants() < 1) {
		return false;
	}
	m_activePlants->selectByType();
	return true;
}

bool Forest::selectPlants(const Ray & ray, SelectionContext::SelectMode mode)
{
	if(numPlants() < 1) return false;
	
	if(!intersectGround(ray) ) {
		if(!intersectGrid(ray) ) {
			intersectWorldBox(ray);
			return false;
		}
	}
	
	m_activePlants->setCenter(m_intersectCtx.m_hitP, m_intersectCtx.m_hitN);
	m_activePlants->select(mode);
	
	return true;
}

bool Forest::closeToOccupiedPosition(CollisionContext * ctx)
{
	sdb::Coord3 c0 = m_grid->gridCoord((const float *)&ctx->_pos);
	ForestCell * cell = m_grid->findCell(c0);
	if(testNeighborsInCell(ctx, cell) ) return true;
	
	BoundingBox b = m_grid->coordToGridBBox(c0);
	
	sdb::Coord3 c1 = c0;
	if(ctx->getXMin() < b.getMin(0) ) {
		 c1.x = c0.x - 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(ctx, cell) ) return true;
	}
	if(ctx->getXMax() > b.getMax(0) ) {
		 c1.x = c0.x + 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(ctx, cell) ) return true;
	}
	c1.x = c0.x;
	if(ctx->getYMin() < b.getMin(1) ) {
		 c1.y = c0.y - 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(ctx, cell) ) return true;
	}
	if(ctx->getYMax() > b.getMax(1) ) {
		 c1.y = c0.y + 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(ctx, cell) ) return true;
	}
	c1.y = c0.y;
	if(ctx->getZMin() < b.getMin(2) ) {
		 c1.z = c0.z - 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(ctx, cell) ) return true;
	}
	if(ctx->getZMax() > b.getMax(2) ) {
		 c1.z = c0.z + 1;
		 cell = m_grid->findCell(c1);
		 if(testNeighborsInCell(ctx, cell) ) return true;
	}
	return false;
}

bool Forest::testNeighborsInCell(CollisionContext * ctx,
					ForestCell * cell)
{
	if(!cell) {
		return false;
	}
	
	if(cell->isEmpty() ) {
		return false;
	}
	
	bool doCollide;
	
	cell->begin();
	while(!cell->end()) {
		PlantData * d = cell->value()->index;
		if(d == NULL) {
			throw "Forest testNeighborsInCell null data";
		}
		
		if(ctx->_minIndex > -1) {
			doCollide = cell->key().x < ctx->_minIndex;
		} else {
			doCollide = true;
		}
		
		if(doCollide) {
			float scale = d->t1->getSide().length() * .5f;
			if(ctx->contact(d->t1->getTranslation(),
							plantSize(cell->key().y) * scale) ) {
				return true;
			}
		}
		
		cell->next();
	}
	return false;
}

const float & Forest::plantSize(const int & idx)
{ return m_examples[idx]->geomSize(); }

ForestGrid * Forest::grid()
{ return m_grid; }

const int & Forest::numActivePlants() const
{ return m_activePlants->numSelected(); }

KdNTree<cvx::Triangle, KdNode4 > * Forest::ground()
{ return m_ground; }

const KdNTree<cvx::Triangle, KdNode4 > * Forest::ground() const
{ return m_ground; }

IntersectionContext * Forest::intersection()
{ return &m_intersectCtx; }

PlantSelection * Forest::selection()
{ return m_activePlants; }

PlantSelection::SelectionTyp * Forest::activePlants()
{ return m_activePlants->data(); }

void Forest::clearAllPlants()
{
	m_activePlants->deselect();
	m_grid->clearPlants();
	m_plants.clear();
    
	std::vector<PlantData *>::iterator itb = m_pool.begin();
	for(;itb!=m_pool.end();++itb) {
		delete (*itb)->t1;
		delete (*itb)->t2;
		delete (*itb);
	}
    m_pool.clear();
	m_numPlants = 0;
	
}

int Forest::getBindPoint(Vector3F & pos, GroundBind * bind)
{
	int geom, component;
	bind->getGeomComp(geom, component);
	if(geom < 0 || geom > 999) return -1;
	if(geom >= numGroundMeshes() ) return 0;
	
	ATriangleMesh * mesh = m_groundMeshes[geom];
	if(component < 0 || component >= mesh->numTriangles() ) return 0;
	unsigned * tri = mesh->triangleIndices(component);
	Vector3F * pnt = mesh->points();
	pos = pnt[tri[0]] * bind->m_w0
			+ pnt[tri[1]] * bind->m_w1
			+ pnt[tri[2]] * bind->m_w2;
	pos += bind->m_offset;
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

Vector3F Forest::bindNormal(const GroundBind * bind) const
{
    int igeom, icomp;
    bind->getGeomComp(igeom, icomp);
    if(igeom >= numGroundMeshes() ) {
        throw "Forest bindNormal igeom out of range";   
    }
    
    const ATriangleMesh * msh = getGroundMesh(igeom);
    if(icomp >= msh->numTriangles() ) {
        throw "Forest bindNormal icomp out of range";   
    }
    
    return msh->triangleNormal(icomp);
}

void Forest::getClosestBind(GroundBind * bind) const
{
	bind->setGeomComp(m_closestPointTest._igeometry, 
								m_closestPointTest._icomponent );
	bind->m_w0 = m_closestPointTest._contributes[0];
	bind->m_w1 = m_closestPointTest._contributes[1];
	bind->m_w2 = m_closestPointTest._contributes[2];
}

void Forest::getBindTexcoord(Float2 & dst) const
{
	const ATriangleMesh * mesh = getGroundMesh(m_closestPointTest._igeometry);
	if(!mesh) {
		return;
	}
	
	const Float2 * triuvs = mesh->triangleTexcoord(m_closestPointTest._icomponent);
	dst.x = triuvs[0].x * m_closestPointTest._contributes[0]
			+ triuvs[1].x * m_closestPointTest._contributes[1]
			+ triuvs[2].x * m_closestPointTest._contributes[2];
	dst.y = triuvs[0].y * m_closestPointTest._contributes[0]
			+ triuvs[1].y * m_closestPointTest._contributes[1]
			+ triuvs[2].y * m_closestPointTest._contributes[2];
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
	try {
	engine.intersect<cvx::Triangle, KdNode4>(m_ground, &m_intersectCtx );
	} catch(const char * ex) {
	    std::cerr<<"Forest intersectGround caught: "<<ex;
	} catch(...) {
	    std::cerr<<"Forest intersectGround caught something";
	}
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
	m_pool.push_back(d);
	
	Plant * p = new Plant;
	p->key = sdb::Coord2(m_plants.size(), plantTypeId);
	p->index = m_pool.back();
	m_plants.push_back(p);
	m_lastPlantInd = p->key.x;
	
	const Vector3F & at = tm.getTranslation();
	
	try {
		m_grid->insert((const float *)&at, p );
	} catch (const char * ex) {
		std::cerr<<"forest add plant caught: "<<ex;
		return;
	}
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
    if(!m_ground) {
		return true;
	}
    return m_ground->isEmpty();
}

int Forest::numPlantExamples() const
{ return m_examples.size(); }

void Forest::addPlantExample(ExampVox * x, const int & islot)
{
	if(m_exampleIndices.find(x) != m_exampleIndices.end() ) {
		return;
	}
	
	std::cout<<"\n add example "<<islot;
	
	m_exampleIndices[x] = m_examples.size();
	m_examples[islot] = x;
	const int ne = x->numExamples();
	if(ne > 1) {
		for(int i=0;i<ne;++i) {
			int elmi = plant::exampleIndex(islot, i);
			m_examples[elmi] = x->getExample(i);
		}
	}
	
}

ExampVox * Forest::plantExample(const int & idx)
{
	if(m_examples.find(idx) == m_examples.end() ) {
		return NULL;
	}
	return m_examples[idx]; 
}

void Forest::setSelectTypeFilter(int flt)
{ m_activePlants->setTypeFilter(flt); }

std::string Forest::groundBuildLog() const
{ 
    if(!m_ground) {
		return " error ground Kdtree not built"; 
	}
    return "";//m_ground->buildLog();
}

const sdb::VectorArray<cvx::Triangle> & Forest::triangles() const
{ return m_triangles; }

const float & Forest::gridSize() const
{ return m_grid->gridSize(); }

void Forest::onSampleChanged()
{
	std::cout<<"Forest on sample changed";
    std::cout.flush();
	updateGrid();
	countNumSamples();
}

void Forest::onPlantChanged()
{
	std::cout<<"Forest on plant changed";
    std::cout.flush();
	updateGrid();
	countNumPlants();
}

void Forest::intersectWorldBox(const Ray & ray)
{
	float h0, h1;
	if(!gridBoundingBox().intersect(ray, &h0, &h1) ) {
		h1 = 1e6f;
	}
	m_intersectCtx.m_hitP = ray.travel(h1);
	m_intersectCtx.m_hitN = ray.m_dir;
}

bool Forest::closeToOccupiedBundlePosition(CollisionContext * ctx)
{
	sdb::Coord3 c0 = m_grid->gridCoord((const float *)&ctx->_pos);
	ForestCell * cell = m_grid->findCell(c0);
	if(!cell) {
		return false;
	}
	
	if(cell->isEmpty() ) {
		return false;
	}
	
	Vector3F pos1;
	float size1;
	
	cell->begin();
	while(!cell->end()) {
		PlantData * d = cell->value()->index;
		if(d == NULL) {
			throw "Forest testNeighborsInCell null data";
		}
		
		const int k = cell->key().y;
		if(plant::bundleIndex(k) == ctx->_bundleIndex) {
		
			size1 = d->t1->getSide().length() * ctx->_bundleScaling * .5f;
			pos1 = d->t1->getTranslation() - d->t2->m_offset;
			
		} else {
			size1 = d->t1->getSide().length() * plantSize(k) * .5f;
			pos1 = d->t1->getTranslation();
			
		}
		
		if(ctx->contact(pos1, size1) ) {
				return true;
		}
		
		cell->next();
	}
	
	return false;
}

const int & Forest::lastPlantIndex() const
{ return m_lastPlantInd; }

void Forest::countNumSamples()
{ grid()->countActiveSamples(); }

const int & Forest::sampleLevel() const
{ return m_sampleFlt->maxSampleLevel(); }

void Forest::reshuffleSamples()
{
	if(!m_grid) return;
	if(!m_ground) return;
	if(m_ground->isEmpty() ) return;
	
	m_grid->reshuffleSamples<SampleFilter>(*m_sampleFlt);
}

void Forest::processSampleFilter()
{
	if(!m_grid) return;
	m_grid->processFilter<SampleFilter>(*m_sampleFlt);
	countNumSamples();
}
	
const float & Forest::filterPortion() const
{
	return m_sampleFlt->portion();
}

void Forest::setFilterPortion(const float & x)
{ return m_sampleFlt->setPortion(x); }

void Forest::setFilterNoise(const ANoise3Sampler & param)
{
	m_sampleFlt->m_noiseOrigin = param.m_noiseOrigin;
	m_sampleFlt->m_noiseFrequency = 3.1f * param.m_noiseFrequency / gridSize();
	m_sampleFlt->m_noiseLacunarity = param.m_noiseLacunarity;
	m_sampleFlt->m_noiseLevel = param.m_noiseLevel;
	m_sampleFlt->m_noiseGain = param.m_noiseGain;
	m_sampleFlt->m_noiseOctave = param.m_noiseOctave;
}

void Forest::setFilterImage(const ExrImage * img)
{
    m_sampleFlt->m_imageSampler = img;
}

void Forest::deselectSamples()
{
	ForestGrid * g = grid();
	if(g) {
		g->deselectCells();
	}
}

int Forest::numVisibleSamples()
{
	return grid()->numVisibleSamples();
}

bool Forest::selectGroundSamples(const Ray & ray, SelectionContext::SelectMode mode)
{
	if(!intersectGround(ray) ) {
		intersectWorldBox(ray);
		return false;
	}
	
	typedef IntersectEngine<cvx::Triangle, KdNode4 > FIntersectTyp;
	FIntersectTyp ineng(m_ground);
    
	typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
    FClosestTyp clseng(m_ground);
	
	m_sampleFlt->setMode(mode );
	
	cvx::Sphere sph;
	sph.set(m_intersectCtx.m_hitP, m_activePlants->radius() );
	
	m_grid->selectCells<FIntersectTyp, FClosestTyp, SampleFilter, cvx::Sphere >(ineng, clseng,
					*m_sampleFlt, sph );
	return true;
}

bool Forest::selectGroundSamples(const AFrustum & fru, SelectionContext::SelectMode mode)
{
	typedef IntersectEngine<cvx::Triangle, KdNode4 > FIntersectTyp;
	FIntersectTyp ineng(m_ground);
    
	typedef ClosestToPointEngine<cvx::Triangle, KdNode4 > FClosestTyp;
    FClosestTyp clseng(m_ground);
	
	m_sampleFlt->setMode(mode );
	
	m_grid->selectCells<FIntersectTyp, FClosestTyp, SampleFilter, AFrustum >(ineng, clseng,
					*m_sampleFlt, fru );
					
	return true;
}

void Forest::getStatistics(std::map<std::string, std::string > & stats)
{
	std::stringstream sst;
	const int & nbx = numPlants();
	sst<<nbx;
	stats["box"] = sst.str();
	
	sst.str("");
	int nc = numCells();
	sst<<nc;
	stats["cell"] = sst.str();
	
	sst.str("");
	sst<<gridBoundingBox();
	stats["bound"] = sst.str();
	
	sst.str("");
	int ntri = m_triangles.size();
	sst<<ntri;
	stats["triangle"] = sst.str();
}

void Forest::setFilterPlantTypeMap(const std::vector<int> & indices)
{
	m_sampleFlt->resetPlantTypeIndices(indices);
}

void Forest::setFilterPlantColors(const std::vector<Vector3F> & colors)
{
	m_sampleFlt->resetPlantTypeColors(colors);
}

void Forest::updateSamplePlantType()
{
	grid()->assignSamplePlantType<SampleFilter>(*m_sampleFlt);
}

void Forest::updateSampleColor()
{
	grid()->colorSampleByPlantType<SampleFilter>(*m_sampleFlt);
}

int Forest::randomExampleInd() const
{
	return m_sampleFlt->selectPlantType(rand() & 65535);
}

void Forest::moveSelectionCenter(const Vector3F & dv)
{
	selection()->moveCenter(dv);
	m_intersectCtx.m_hitP += dv;
}

}
