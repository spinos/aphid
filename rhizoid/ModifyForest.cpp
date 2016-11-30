/*
 *  ModifyForest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ModifyForest.h"
#include <TriangleRaster.h>
#include <BarycentricCoordinate.h>
#include <ANoise3.h>
#include <../qtogl/ebp.h>

namespace aphid {

ModifyForest::ModifyForest() 
{ 
	m_raster = new TriangleRaster;
	m_bary = new BarycentricCoordinate;
	m_seed = rand() % 999999; 
    m_noiseWeight = 0.f;
    m_ebpSampler = new EbpGrid;
}

ModifyForest::~ModifyForest() 
{
	delete m_raster;
	delete m_bary;
}

void ModifyForest::setNoiseWeight(float x)
{ m_noiseWeight = x; } 

bool ModifyForest::growOnGround(GrowOption & option)
{
	if(numActiveGroundFaces() < 1) return false;
	sdb::Sequence<int> * prims = activeGround()->primIndices();
	const sdb::VectorArray<cvx::Triangle> & tris = triangles();
	
	prims->begin();
	while(!prims->end() ) {
	
		const cvx::Triangle * t = tris[prims->key() ];
		growOnFace(t->ind0(), t->ind1(), option);
		
		prims->next();
	}
	
    return true;
}

void ModifyForest::growOnFace(const int & geoId, const int & triId,
					GrowOption & option)
{
	GroundBind bind;
	const ATriangleMesh * mesh = groundMeshes()[geoId];
	Vector3F * p = mesh->points();
	if(option.m_alongNormal)
		option.m_upDirection = mesh->triangleNormal(triId );
		
	unsigned * tri = mesh->triangleIndices(triId );
	
	if(m_raster->create(p[tri[0]], p[tri[1]], p[tri[2]] ) ) {
		m_bary->create(p[tri[0]], p[tri[1]], p[tri[2]] );
		
		bind.setGeomComp(geoId, triId );
		growOnTriangle(m_raster, m_bary, bind, option);
	}
}

void ModifyForest::growOnFaces(Geometry * geo, sdb::Sequence<unsigned> * components, 
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
		
		if(m_raster->create(p[tri[0]], p[tri[1]], p[tri[2]] ) ) {
			m_bary->create(p[tri[0]], p[tri[1]], p[tri[2]] );
			
			bind.setGeomComp(geoId, components->key() );
			growOnTriangle(m_raster, m_bary, bind, option);
		}
		components->next();
	}
}

void ModifyForest::growOnTriangle(TriangleRaster * tri, 
							BarycentricCoordinate * bar,
							GroundBind & bind,
							GrowOption & option)
{
	float sampleSize = plantSize(option.m_plantId) * .67f;
	int grid_x, grid_y;
	tri->gridSize(sampleSize, grid_x, grid_y);
	int numSamples = grid_x * grid_y;
	
	const bool limitRadius = option.m_radius > 1e-3f;
	Vector3F *samples = new Vector3F[numSamples];
	char *hits = new char[numSamples];
	tri->genSamples(sampleSize, grid_x, grid_y, samples, hits);
	float scale;
	Matrix44F tm;
/// relative to grid
	const float freq = option.m_noiseFrequency / (gridSize() + 1e-3f);
	for(int s = 0; s < numSamples; s++) {
		if(!hits[s]) continue;	
	
		Vector3F & pos = samples[s];
		
/// limited by noise level
		if(option.m_noiseLevel > 0.001f) {
			if(ANoise3::FractalF((const float *)&pos,
							(const float *)&option.m_noiseOrigin,
							freq,
							option.m_noiseLacunarity,
							option.m_noiseOctave,
							option.m_noiseGain ) < option.m_noiseLevel )
				continue;
		}
		
		if(limitRadius) {
			if(pos.distanceTo(option.m_centerPoint) >  option.m_radius)
				continue;
		}
			
		randomSpaceAt(pos, option, tm, scale);
		float scaledSize = sampleSize * 1.5f * scale;
		float limitedMargin = getNoise2(option.m_minMarginSize, option.m_maxMarginSize);
/// limit low margin
		if(limitedMargin < -.99f * scaledSize) limitedMargin = -.99f * scaledSize;
		float delta = limitedMargin + scaledSize;
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

bool ModifyForest::growAt(const Ray & ray, GrowOption & option)
{
	if(!intersectGround(ray) ) return false;
	
	IntersectionContext * ctx = intersection();
	
	if(option.m_multiGrow) {
		activeGround()->deselect();
		selectGroundFaces(ray, SelectionContext::Append);
/// limit radius
		option.m_centerPoint = ctx->m_hitP;
		option.m_radius = selection()->radius();
		growOnGround(option);
		return true;
	}
	
/// ind to source
	if(ctx->m_componentIdx >= ground()->primIndirection().size() ) {
		std::cout<<"\n oor component idx "<<ctx->m_componentIdx
			<<" >= "<<ground()->primIndirection().size();
		return false;
	}
/// idx of source
    const cvx::Triangle * t = ground()->source()->get(ctx->m_componentIdx);
	
/// ind to geom	
	if(t->ind0() >= groundMeshes().size() ) {
		std::cout<<"\n oor mesh idx "<<t->ind0()
			<<" >= "<<groundMeshes().size();
		return false;
	}
	
	if(option.m_alongNormal)
		option.m_upDirection = t->calculateNormal();
    
	if(!m_raster->create(t->P(0), t->P(1), t->P(2) ) ) return false;
	
	m_bary->create(t->P(0), t->P(1), t->P(2) );
    
	GroundBind bind;
	bind.setGeomComp(t->ind0(), t->ind1() );
		Matrix44F tm;
		float scale;
		randomSpaceAt(ctx->m_hitP, option, tm, scale); 
		float scaledSize = plantSize(option.m_plantId) * scale;
		float limitedMargin = getNoise2(option.m_minMarginSize, option.m_maxMarginSize);
/// limit low margin
		if(limitedMargin < -.99f * scaledSize) limitedMargin = -.99f * scaledSize;
		float delta = limitedMargin + scaledSize;
		if(closeToOccupiedPosition(ctx->m_hitP, delta)) return false;
		
		m_bary->project(ctx->m_hitP);
		m_bary->compute();
		if(!m_bary->insideTriangle()) {
			std::cout<<"\n out of triangle pnt "<<ctx->m_hitP;
            return false;
        }
        
		bind.m_w0 = m_bary->getV(0);
		bind.m_w1 = m_bary->getV(1);
		bind.m_w2 = m_bary->getV(2);
		
		addPlant(tm, bind, option.m_plantId);
	
    return true;
}

bool ModifyForest::growAt(const Matrix44F & trans, GrowOption & option)
{        
	GroundBind bind;
	if(option.m_stickToGround) {
	Vector3F pog;
	if(!bindToGround(&bind, trans.getTranslation(), pog) )
		return false;
	
/// x size
	float scale = trans.getSide().length();
	float scaledSize = plantSize(option.m_plantId) * scale;
	if(closeToOccupiedPosition(pog, scaledSize)) 
		return false;
        
	Matrix44F tm = trans;
/// snap to ground
	tm.setTranslation(pog);
	addPlant(tm, bind, option.m_plantId);
	}
	else {
/// disable ground binding
		bind.setGeomComp(1023, 0);
		addPlant(trans, bind, option.m_plantId);
	}
	
	return true;
}

void ModifyForest::replaceAt(const Ray & ray, GrowOption & option)
{
	if(!calculateSelecedWeight(ray)) return;
	
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			if(m_pnoise.rfloat(m_seed) < option.m_strength) {
				Plant * pl = arr->value()->m_reference;
				*pl->index->t3 = option.m_plantId;
			}
			m_seed++;
		}
		arr->next();
	}
}

void ModifyForest::clearPlant(Plant * pl, int k)
{
	const Vector3F & pos = pl->index->t1->getTranslation();
	sdb::Coord3 c0 = grid()->gridCoord((const float *)&pos);
	sdb::Array<int, Plant> * cell = grid()->findCell(c0 );
	if(cell) {
		cell->remove(k );
		if(cell->isEmpty() )
			grid()->remove(c0);
	}
}

void ModifyForest::clearSelected()
{
	std::vector<int> idToClear;
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		
		idToClear.push_back(arr->key() );
		Plant * pl = arr->value()->m_reference;
		clearPlant(pl, arr->key() );
			
		arr->next();
	}
	
	std::vector<int>::const_iterator it = idToClear.begin();
	for(;it!=idToClear.end();++it) {
		arr->remove(*it);
	}
	idToClear.clear();
}

void ModifyForest::clearAt(const Ray & ray, GrowOption & option)
{
	if(!calculateSelecedWeight(ray)) return;
	
	std::vector<int> idToClear;
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			if(m_pnoise.rfloat(m_seed) < option.m_strength) {
				idToClear.push_back(arr->key() );
				Plant * pl = arr->value()->m_reference;
				clearPlant(pl, arr->key() );
			}
			m_seed++;
		}
		arr->next();
	}
	
	std::vector<int>::const_iterator it = idToClear.begin();
	for(;it!=idToClear.end();++it) {
		arr->remove(*it);
	}
	idToClear.clear();
}

void ModifyForest::scaleBrushAt(const Ray & ray, float magnitude)
{
    if(!intersectGround(ray)) {
		if(!intersectGrid(ray)) return;
	}
	
	IntersectionContext * ctx = intersection();
	
	selection()->setCenter(ctx->m_hitP, ctx->m_hitN);
	
    float r = selection()->radius();
    r *= 1.f + magnitude;
    selection()->setRadius(r);
}

void ModifyForest::scaleAt(const Ray & ray, float magnitude)
{
    if(!calculateSelecedWeight(ray)) return;
    
    sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			Matrix44F * mat = plantd->t1;
            mat->scaleBy(1.f + magnitude * wei * (1.f + getNoise() ) );
		}
		arr->next();
	}
}

void ModifyForest::movePlant(GrowOption & option)
{
	Vector3F tv, pos, bindPos;
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		tv.x = getNoise2(-option.m_maxMarginSize, option.m_maxMarginSize);
		tv.y = getNoise2(-option.m_maxMarginSize, option.m_maxMarginSize);
		tv.z = getNoise2(-option.m_maxMarginSize, option.m_maxMarginSize);
		
		PlantData * plantd = arr->value()->m_reference->index;
		pos = plantd->t1->getTranslation() + tv;
		
		bindToGround(plantd, pos, bindPos);
			
		plantd->t1->setTranslation(bindPos );
		displacePlantInGrid(arr->value() );
		
		arr->next();
	}
}

void ModifyForest::rotatePlant(GrowOption & option)
{
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		Matrix44F * mat = arr->value()->m_reference->index->t1;
		
		Vector3F vx(mat->M(0, 0), mat->M(0, 1), mat->M(0, 2));
		Vector3F vy(mat->M(1, 0), mat->M(1, 1), mat->M(1, 2));
		Vector3F vz(mat->M(2, 0), mat->M(2, 1), mat->M(2, 2));
		
		float sx = vx.length();
		float sy = vy.length();
		float sz = vz.length();
		
		vx.normalize();
		vx.x += getNoise2(-option.m_rotateNoise, option.m_rotateNoise);
		vx.y += getNoise2(-option.m_rotateNoise, option.m_rotateNoise);
		vx.z += getNoise2(-option.m_rotateNoise, option.m_rotateNoise);
		vx.normalize();
		
		vz = vx.cross(vy);
		vz.normalize();
		
		vy = vz.cross(vx);
		vy.normalize();
		
		vx *= sx;
		vy *= sy;
		vz *= sz;
		
		*mat->m(0, 0) = vx.x;
		*mat->m(0, 1) = vx.y;
		*mat->m(0, 2) = vx.z;
		*mat->m(1, 0) = vy.x;
		*mat->m(1, 1) = vy.y;
		*mat->m(1, 2) = vy.z;
		*mat->m(2, 0) = vz.x;
		*mat->m(2, 1) = vz.y;
		*mat->m(2, 2) = vz.z;
		
		arr->next();
	}
}

void ModifyForest::scalePlant(GrowOption & option)
{
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		PlantData * plantd = arr->value()->m_reference->index;
		Matrix44F * mat = plantd->t1;
		float rat = getNoise2(option.m_minScale, option.m_maxScale) / mat->getSide().length();
		if(rat < .1f) rat = .1f;
		mat->scaleBy(rat );
		arr->next();
	}
}

void ModifyForest::rotateAt(const Ray & ray, float magnitude, int axis)
{
    if(!calculateSelecedWeight(ray)) return;
    Vector3F first, second, third;
    float scaling;
    sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			Matrix44F * tm = plantd->t1;
            
            first.set(tm->M(0,0), tm->M(0,1), tm->M(0,2));
			second.set(tm->M(1,0), tm->M(1,1), tm->M(1,2));
			third.set(tm->M(2,0), tm->M(2,1), tm->M(2,2));
            
			scaling = first.length();
			first.normalize();
			second.normalize();
			third.normalize();
			if(axis == 0) {
				second.rotateAroundAxis(first, magnitude * wei * (1.f + getNoise() ) );
				third = first.cross(second);				
			}
			else if(axis == 1) {
				first.rotateAroundAxis(second, magnitude * wei * (1.f + getNoise() ) );
				third = first.cross(second);
			}
			else {
				first.rotateAroundAxis(third, magnitude * wei * (1.f + getNoise() ) );
				second = third.cross(first);				
			}
			
			first.normalize();
			second.normalize();
			third.normalize();
			first *= scaling;
			second *= scaling;
			third *= scaling;
			
			*tm->m(0, 0) = first.x;
			*tm->m(0, 1) = first.y;
			*tm->m(0, 2) = first.z;
			*tm->m(1, 0) = second.x;
			*tm->m(1, 1) = second.y;
			*tm->m(1, 2) = second.z;
			*tm->m(2, 0) = third.x;
			*tm->m(2, 1) = third.y;
			*tm->m(2, 2) = third.z;
		}
		arr->next();
	}
}

void ModifyForest::movePlant(const Ray & ray,
						const Vector3F & displaceNear, const Vector3F & displaceFar,
						const float & clipNear, const float & clipFar)
{
	if(!calculateSelecedWeight(ray)) return;
	
	IntersectionContext * ctx = intersection();
	
	const float depth = ctx->m_hitP.distanceTo(ray.m_origin);
	Vector3F disp = displaceNear 
					+ (displaceFar - displaceNear) * depth / (clipFar-clipNear);
	if(disp.length() < .1f) return;
    disp *= .5f;

	Vector3F pos, bindPos;
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-3f) { 
            
			PlantData * plantd = arr->value()->m_reference->index;
			pos = plantd->t1->getTranslation() + disp * wei * (1.f + getNoise() );
			
			bindToGround(plantd, pos, bindPos);
			
			plantd->t1->setTranslation(bindPos );
			displacePlantInGrid(arr->value() );
			
		}
		arr->next();
	}
}

void ModifyForest::rotatePlant(const Ray & ray,
					const Vector3F & displaceNear, const Vector3F & displaceFar,
					const float & clipNear, const float & clipFar)
{
	if(!calculateSelecedWeight(ray)) return;
	
	IntersectionContext * ctx = intersection();
	
	const float depth = ctx->m_hitP.distanceTo(ray.m_origin);
    Vector3F disp = displaceNear 
					+ (displaceFar - displaceNear) * depth / (clipFar-clipNear);
	if(disp.length() < .1f) return;
    disp.normalize();
    disp.x *= .15f;
    disp.y *= .5f;
    disp.z *= .15f;
    //std::cout<<"\n disp rot"<<disp;
    //std::cout.flush();
	
	Vector3F pside, pup, pfront, vscale;
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-3f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			
            pside = plantd->t1->getSide();
            pup = plantd->t1->getUp();
            pfront = plantd->t1->getFront();
            vscale.set(pside.length(), pup.length(), pfront.length() );
            
            pside.normalize();
            pup.normalize();
            
            pup += disp * wei * (1.f + getNoise() );
            pup.normalize();
            
            pfront = pside.cross(pup);
            
            pside = pup.cross(pfront);
            pside.normalize();
            
            plantd->t1->setOrientations(pside * vscale.x, 
                                        pup * vscale.y, 
                                        pfront * vscale.z);
		}
		arr->next();
	}
}

void ModifyForest::moveWithGround()
{
	activeGround()->deselect();
	selection()->deselect();
	if(numPlants() < 1) return;
	if(isGroundEmpty() ) return;
	
	grid()->begin();
	while(!grid()->end() ) {
		movePlantsWithGround(grid()->value() );
		grid()->next();
	}
	
	sdb::Array<int, PlantInstance> * arr = activePlants();
	if(arr->size() < 1) return;
	
	arr->begin();
	while(!arr->end() ) {
		displacePlantInGrid(arr->value() );
		arr->next();
	}
	selection()->deselect();
	updateGrid();
}

void ModifyForest::movePlantsWithGround(sdb::Array<int, Plant> * arr)
{
	Vector3F bindP, curP;
	arr->begin();
	while(!arr->end() ) {
		Plant * pl = arr->value();
		curP = pl->index->t1->getTranslation();
		GroundBind * bind = pl->index->t2;
		const int bindStat = getBindPoint(bindP, bind);
		if(bindStat < 0) {
/// disabled, use current position
			bindP = curP;
		}
		else if(bindStat < 1 ) {
			bindToGround(pl->index, curP, bindP);
		}
		
		if(curP.distance2To(bindP) > 1e-5f) {
/// select for later displace
			selection()->select(pl);
/// relocate
			pl->index->t1->setTranslation(bindP);
		}
			
		arr->next();
	}
}

void ModifyForest::randomSpaceAt(const Vector3F & pos, const GrowOption & option,
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

bool ModifyForest::calculateSelecedWeight(const Ray & ray)
{
    if(numActivePlants() < 1 ) return false;
	if(!intersectGround(ray) ) {
		if(!intersectGrid(ray)) return false;
	}
	
	IntersectionContext * ctx = intersection();
	
	selection()->setCenter(ctx->m_hitP, ctx->m_hitN);
	selection()->calculateWeight();
    return true;
}

float ModifyForest::getNoise() const
{ return m_noiseWeight * (float(rand()%991) / 991.f - .5f); }

float ModifyForest::getNoise2(const float & a, const float & b) const
{ return a + (b - a) * (float(rand()%991) ) / 991.f; }

void ModifyForest::erectActive()
{
	if(numActivePlants() < 1) return;
	
	Vector3F worldUp(0.f, 1.f, 0.f);	
	sdb::Array<int, PlantInstance> * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		Matrix44F * mat = arr->value()->m_reference->index->t1;
		
		Vector3F vx(mat->M(0, 0), mat->M(0, 1), mat->M(0, 2));
		Vector3F vy(mat->M(1, 0), mat->M(1, 1), mat->M(1, 2));
		Vector3F vz(mat->M(2, 0), mat->M(2, 1), mat->M(2, 2));
		
		float sx = vx.length();
		float sy = vy.length();
		float sz = vz.length();
		
		vx.y = 0.f;
		vx.normalize();
		
		vz = vx.cross(worldUp);
		vz.normalize();
		
		vx *= sx;
		vy = Vector3F(0.f, sy, 0.f);
		vz *= sz;
		
		*mat->m(0, 0) = vx.x;
		*mat->m(0, 1) = vx.y;
		*mat->m(0, 2) = vx.z;
		*mat->m(1, 0) = vy.x;
		*mat->m(1, 1) = vy.y;
		*mat->m(1, 2) = vy.z;
		*mat->m(2, 0) = vz.x;
		*mat->m(2, 1) = vz.y;
		*mat->m(2, 2) = vz.z;
		
		arr->next();
	}
}

void ModifyForest::removeTypedPlants(int x)
{
	if(numPlantExamples() < 1) {
		removeAllPlants();
		return;
	}
	selection()->deselect();
	selectTypedPlants(x);
	std::cout<<"\n remove "<<numActivePlants()<<" type"<<x<<" plants";
	std::cout.flush();
	clearSelected();
}

void ModifyForest::finishGroundSelection(GrowOption & option)
{
    if(numActiveGroundFaces() < 1) return;
    const float gz = 2.f * (plantSize(option.m_plantId) + option.m_minMarginSize + option.m_maxMarginSize);
    std::cout<<"\n sample ground by "<<gz;
    
typedef PrimInd<sdb::Sequence<int>, sdb::VectorArray<cvx::Triangle >, cvx::Triangle > TIntersect;
	TIntersect fintersect(activeGround()->primIndices(), 
	                        &triangles() );
	std::cout<<"\n bbox "<<fintersect;
	m_ebpSampler->fillBox(fintersect, gz * 8.f);
	m_ebpSampler->subdivideToLevel<TIntersect>(fintersect, 0, 3);
	m_ebpSampler->insertNodeAtLevel(3);
	m_ebpSampler->cachePositions();
	int numParticles = m_ebpSampler->numParticles();
	std::cout<<"\n num cells "<<m_ebpSampler->numCellsAtLevel(3)
	    <<"\n num instances "<<numParticles;
	m_ebpSampler->calculateBBox();
	std::cout<<"\n box sample "<<m_ebpSampler->boundingBox();
	std::cout.flush();
	for(int i=0;i<20;++i) {
		m_ebpSampler->update();    
	}
}

}