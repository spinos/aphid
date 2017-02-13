/*
 *  ModifyForest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ModifyForest.h"
#include <geom/ATriangleMesh.h>
#include <math/ANoise3.h>
#include <sdb/ebp.h>
#include "ExampVox.h"
#include <ForestCell.h>
#include <geom/PrimInd.h>
#include <ctime>

namespace aphid {

ModifyForest::ModifyForest()
{ 
	m_manipulateMode = manNone;
	m_bary = new BarycentricCoordinate;
	m_pnoise = new PseudoNoise;
	m_seed = rand() % 999999; 
    m_noiseWeight = 0.f;
	std::time_t tim(NULL);
	srand((int)tim);
}

ModifyForest::~ModifyForest() 
{
	delete m_bary;
	delete m_pnoise;
}

void ModifyForest::setNoiseWeight(float x)
{ m_noiseWeight = x; }

bool ModifyForest::growOnGround(GrowOption & option)
{	
    if(numActiveGroundFaces() < 1) {
		std::cout<<"\n ModifyForest has no active ground to sample";
		return false;
	}
	
    EbpGrid ebpSampler;
	if(!sampleGround(&ebpSampler, option))
		return false;
		
	const Vector3F * psamp = ebpSampler.positions();
	const float freq = option.m_noiseFrequency / (gridSize() + 1e-3f);
	const bool limitRadius = option.m_radius > 1e-3f;
   
	GroundBind bind;
	float scale;
	Matrix44F tm;
	Vector3F pog;
	CollisionContext collctx;
	collctx._minIndex = -1;
	Float2 sampleTexCoord;
	
	try {
	for(int i=0;i<ebpSampler.numParticles();++i) {
/// tree division may not satify distance condition
/// on the safe side to start at very large distance
		if(!closestPointOnGround(pog, psamp[i], 1e7f) ) {
		    std::cout<<"failed closestPointOnGround test"<<psamp[i];
		    continue;
		}
		
		setBind(&bind);
		getBindTexcoord(sampleTexCoord);
	
		if(option.m_noiseLevel > 1e-3f) {
			if(ANoise3::FractalF((const float *)&pog,
							(const float *)&option.m_noiseOrigin,
							freq,
							option.m_noiseLacunarity,
							option.m_noiseOctave,
							option.m_noiseGain ) < option.m_noiseLevel ) {
				continue;
			}
		}
		
		if(limitRadius) {
			if(pog.distanceTo(option.m_centerPoint) >  option.m_radius) {
			    continue;
			}
		}
		
		if(option.m_alongNormal) {
			option.m_upDirection = bindNormal(&bind);
		}
		
		randomSpaceAt(pog, option, tm, scale);
		
		const ExampVox * v = plantExample(option.m_plantId);
		
		collctx._radius = v->geomSize() * scale * .5f;
		collctx._minDistance = option.m_minMarginSize;
		collctx._maxDistance = option.m_maxMarginSize;
		collctx._pos = pog;
		if(v->numExamples() > 1) {
			growBundle(option, bind, v, option.m_plantId, tm, &collctx);			
		
		} else {
			growSingle(option, bind, option.m_plantId, tm, &collctx);
					
		}

	}
	} catch (const char * ex) {
	    std::cerr<<"ModifyForest growOnGround caught: "<<ex;
	} catch (...) {
	    std::cerr<<"ModifyForest growOnGround caught something";
	}
	
	ebpSampler.clear();
    return true;
}

void ModifyForest::growBundle(GrowOption & option,
				GroundBind & bind,
				const ExampVox * bundle,
				const int & iExample,
				const Matrix44F & tm,
				CollisionContext * collctx)
{	
	IntersectionContext * ctx = intersection();
	const float rayD = bundle->geomExtent() * 4.f;
	collctx->_bundleIndex = iExample;
	collctx->_bundleScaling = bundle->geomSize();
	if(closeToOccupiedBundlePosition(collctx) ) {
		return;
	}
	
	const float scaling = tm.getSide().length();
	
	Matrix44F invTm = tm;
	invTm.inverse();

	const Vector3F vdown(0.f, -1.f, 0.f);
/// world ray dir pointing object down
	Vector3F wdir;
/// instance pos in local space
	Vector3F locP;
/// ground pos offset in local space
	Vector3F groundP;
/// world space instance pos
	Vector3F instP;
	
	collctx->_minIndex = lastPlantIndex();
		
	const int bundleCount = bundle->numInstances();
	for(int i=0;i<bundleCount;++i) {
		const ExampVox::InstanceD & inst = bundle->getInstance(i);
		Matrix44F instTm(inst._trans);
		locP = instTm.getTranslation();
		bind.m_offset = locP;
		locP.y = 0.f;
		
/// to world
		instTm *= tm;
		bind.m_offset = tm.transformAsNormal(bind.m_offset);
		
		instP = instTm.getTranslation();

		wdir = tm.transformAsNormal(vdown);
		wdir.normalize();
		const Ray incident(instP, wdir, 0.f, rayD );
		if(!intersectGround(incident) ) {
			continue;
		}
		
/// to local		
		groundP = invTm.transform(ctx->m_hitP);
		groundP -= locP;
/// to world
		groundP = tm.transformAsNormal(groundP);
/// move in world
		instP += groundP;
		instTm.setTranslation(instP);
		bind.m_offset += groundP;

		const int instExample = plant::exampleIndex(iExample, inst._exampleId );
		
		collctx->_pos = instP;
		collctx->_radius = plantSize(instExample) * scaling * .5f;
		collctx->_minDistance = 0.f;
		collctx->_maxDistance = 0.f;
	
		growSingle(option, bind, instExample, instTm, collctx);
		
	}
}

bool ModifyForest::growSingle(GrowOption & option,
				GroundBind & bind,
				const int & iExample,
				const Matrix44F & tm,
				CollisionContext * collctx)
{
	if(closeToOccupiedPosition(collctx) ) {
		return false;
	}

	addPlant(tm, bind, iExample);
	return true;
}

bool ModifyForest::growAt(const Ray & ray, GrowOption & option)
{
	if(!intersectGround(ray) ) {
		intersectWorldBox(ray);
		return false;
	}
	
	IntersectionContext * ctx = intersection();
	
	if(option.m_multiGrow) {
		activeGround()->deselect();
		selectGroundFaces(ray, SelectionContext::Append);
/// limit radius
		option.m_centerPoint = ctx->m_hitP;
		option.m_radius = selection()->radius();
		return growOnGround(option);
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
	
	if(option.m_alongNormal) {
		option.m_upDirection = t->calculateNormal();
    }
	
	m_bary->create(t->P(0), t->P(1), t->P(2) );
    
	GroundBind bind;
	bind.setGeomComp(t->ind0(), t->ind1() );
	
	m_bary->project(ctx->m_hitP);
	m_bary->compute();
	if(!m_bary->insideTriangle()) {
		std::cout<<"\n out of triangle pnt "<<ctx->m_hitP;
		return false;
	}
	
	bind.m_w0 = m_bary->getV(0);
	bind.m_w1 = m_bary->getV(1);
	bind.m_w2 = m_bary->getV(2);
		
	Matrix44F tm;
	float scale;
	randomSpaceAt(ctx->m_hitP, option, tm, scale); 
	
	const ExampVox * v = plantExample(option.m_plantId);
	
	CollisionContext collctx;
	collctx._minIndex = -1;
	collctx._radius = v->geomSize() * scale * .5f;
	collctx._minDistance = option.m_minMarginSize;
	collctx._maxDistance = option.m_maxMarginSize;
	collctx._pos = ctx->m_hitP;
	
	if(v->numExamples() > 1) {
		growBundle(option, bind, v, option.m_plantId, tm, &collctx);			
	
	} else {
		growSingle(option, bind, option.m_plantId, tm, &collctx);
				
	}
		
    return true;
}

bool ModifyForest::growAt(const Matrix44F & trans, GrowOption & option)
{    
	GroundBind bind;
	Vector3F pog;
	if(!bindToGround(&bind, trans.getTranslation(), pog) ) {
		return false;
	}
	
	if(option.m_alongNormal) {
		option.m_upDirection = bindNormal(&bind);
    }
	
	Matrix44F tm;
	float scale;
/// reshuffle for particles
	if(option.m_isInjectingParticle) {
		randomSpaceAt(pog, option, tm, scale); 
	} else {
/// x size
		scale = trans.getSide().length();
		tm = trans;
	}
	
	const ExampVox * v = plantExample(option.m_plantId);
	
	CollisionContext collctx;
	collctx._minIndex = -1;
	collctx._radius = v->geomSize() * scale * .5f;
	collctx._minDistance = option.m_minMarginSize;
	collctx._maxDistance = option.m_maxMarginSize;
	collctx._pos = pog;
	
	if(option.m_stickToGround) {
/// snap to ground
		bind.m_offset.setZero();
		tm.setTranslation(pog);
	}
	else {
/// preserve the offset
		bind.m_offset = trans.getTranslation() - pog;
		tm = trans;
	}
	
	if(v->numExamples() > 1) {
		growBundle(option, bind, v, option.m_plantId, tm, &collctx);			
	
	} else {
		growSingle(option, bind, option.m_plantId, tm, &collctx);
				
	}
	
	return true;
}

void ModifyForest::replaceAt(const Ray & ray, GrowOption & option)
{
	if(!calculateSelecedWeight(ray)) return;
	
	PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			if(m_pnoise->rfloat(m_seed) < option.m_strength) {
/// todo remove and add
				// Plant * pl = arr->value()->m_reference; 
				// *pl->index->t3 = option.m_plantId;
			}
			m_seed++;
		}
		arr->next();
	}
}

void ModifyForest::clearPlant(Plant * pl, const sdb::Coord2 & k)
{
	const Vector3F & pos = pl->index->t1->getTranslation();
	sdb::Coord3 c0 = grid()->gridCoord((const float *)&pos);
	ForestCell * cell = grid()->findCell(c0 );
	if(cell) {
		cell->remove(k );
		if(cell->isEmpty() )
			grid()->remove(c0);
	}
}

void ModifyForest::clearSelected()
{
	std::vector<sdb::Coord2> idToClear;
	PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		
		idToClear.push_back(arr->key() );
		Plant * pl = arr->value()->m_reference;
		clearPlant(pl, arr->key() );
			
		arr->next();
	}
	
	std::vector<sdb::Coord2>::const_iterator it = idToClear.begin();
	for(;it!=idToClear.end();++it) {
		arr->remove(*it);
	}
	idToClear.clear();
}

void ModifyForest::clearAt(const Ray & ray, GrowOption & option)
{
	if(!calculateSelecedWeight(ray)) return;
	
	std::vector<sdb::Coord2> idToClear;
	PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			if(m_pnoise->rfloat(m_seed) < option.m_strength) {
				idToClear.push_back(arr->key() );
				Plant * pl = arr->value()->m_reference;
				clearPlant(pl, arr->key() );
			}
			m_seed++;
		}
		arr->next();
	}
	
	std::vector<sdb::Coord2>::const_iterator it = idToClear.begin();
	for(;it!=idToClear.end();++it) {
		arr->remove(*it);
	}
	idToClear.clear();
}

void ModifyForest::scaleBrushAt(const Ray & ray, float magnitude)
{
    if(!intersectGround(ray)) {
		if(!intersectGrid(ray)) {
			intersectWorldBox(ray);
			return;
		}
	}
	
	IntersectionContext * ctx = intersection();
	
	selection()->setCenter(ctx->m_hitP, ctx->m_hitN);
	
    float r = selection()->radius();
    r *= 1.f + magnitude;
    selection()->setRadius(r);
}

void ModifyForest::scaleAt(const Ray & ray, float magnitude,
							bool isBundled)
{
    if(!calculateSelecedWeight(ray)) return;
    
    PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			Matrix44F * mat = plantd->t1;
			float scaling = 1.f + magnitude * wei * (1.f + getNoise() );
            mat->scaleBy(scaling );
			if(isBundled && plant::isChildOfBundle(arr->key().y) ) {
				Vector3F & voffset = plantd->t2->m_offset;
				Vector3F psurf = mat->getTranslation() - voffset;
				voffset *= scaling;
				psurf += voffset;
				mat->setTranslation(psurf);
			}
		}
		arr->next();
	}
}

void ModifyForest::movePlant(GrowOption & option)
{
	Vector3F tv, pos, bindPos;
	PlantSelection::SelectionTyp * arr = activePlants();
	try {
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
	} catch (...) {
		std::cerr<<" ModifyForest movePlant caught something ";
	}
}

void ModifyForest::rotatePlant(GrowOption & option)
{
	PlantSelection::SelectionTyp * arr = activePlants();
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
	PlantSelection::SelectionTyp * arr = activePlants();
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
    PlantSelection::SelectionTyp * arr = activePlants();
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
	PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-3f) { 
            
			PlantData * plantd = arr->value()->m_reference->index;
			pos = plantd->t1->getTranslation() + disp * (wei * (1.f + getNoise() ) );
			pos -= plantd->t2->m_offset;
			bindToGround(plantd, pos, bindPos);
			bindPos += plantd->t2->m_offset;
			
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
	PlantSelection::SelectionTyp * arr = activePlants();
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
            
            pup += disp * (wei * (1.f + getNoise() ) );
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

void ModifyForest::resizePlant()
{
	Matrix33F rot;
    Vector3F vdscal, pos, vof;
    float fdscal, lof;
    PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		 float wei = arr->value()->m_weight;
		 if(wei > 1e-3f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			
			Matrix44F * mat = plantd->t1;
			rot = mat->rotation();
			
			getDeltaScaling(vdscal, wei);
			
			fdscal = 1.f;
			
/// scale 3 axises
			if(vdscal.x != 1.f) {
				fdscal = vdscal.x;
			}
			if(vdscal.y != 1.f) {
				fdscal = vdscal.y;
			}
			if(vdscal.z != 1.f) {
				fdscal = vdscal.z;
			}
			
			rot *= fdscal;
			
			mat->setRotation(rot);
			
			vof = plantd->t2->m_offset;
			lof = vof.length();
			if(lof > 1e-2f) {
				pos = mat->getTranslation() - vof;
				vof *= vdscal;
				plantd->t2->m_offset = vof;
				
				mat->setTranslation(pos + vof);
            }
		}
		arr->next();
	}
}

void ModifyForest::translatePlant()
{
    Vector3F pos, dpos, bindPos;
	
    PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		 float wei = arr->value()->m_weight;
		 if(wei > 1e-3f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			
			Matrix44F * mat = plantd->t1;
			pos = mat->getTranslation() - plantd->t2->m_offset;
			
			getDeltaTranslation(dpos, wei);
			pos += dpos;
            bindToGround(plantd, pos, bindPos);
			
			plantd->t1->setTranslation(bindPos + plantd->t2->m_offset );
			displacePlantInGrid(arr->value() );
			
		}
		arr->next();
	}
}

void ModifyForest::rotatePlant()
{
    Matrix33F rot, drot;
    Vector3F pos, vof;
    float lof;
    PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		 float wei = arr->value()->m_weight;
		 if(wei > 1e-3f) { 
			PlantData * plantd = arr->value()->m_reference->index;
			
			Matrix44F * mat = plantd->t1;
			rot = mat->rotation();
			
			getDeltaRotation(drot, wei);
			rot *= drot;
			
			mat->setRotation(rot);
			
			vof = plantd->t2->m_offset;
			lof = vof.length();
			if(lof > 1e-2f) {
				pos = mat->getTranslation() - vof;
				vof = drot.transform(vof);
				plantd->t2->m_offset = vof;
				
				mat->setTranslation(pos + vof);
            }
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
	
	PlantSelection::SelectionTyp * arr = activePlants();
	if(arr->size() < 1) return;
	
	arr->begin();
	while(!arr->end() ) {
		displacePlantInGrid(arr->value() );
		arr->next();
	}
	selection()->deselect();
	updateGrid();
}

void ModifyForest::movePlantsWithGround(ForestCell * arr)
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
	
	scale = (m_pnoise->rfloat(m_seed)) * (option.m_maxScale - option.m_minScale)
				+ option.m_minScale;
	m_seed++;
	
	Vector3F vside(side.x, side.y, side.z);
	Vector3F vaxis(up.x, up.y, up.z);
	float ang = (m_pnoise->rfloat(m_seed) - 0.5f) * option.m_rotateNoise * 2.f * 3.14f;
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
		if(!intersectGrid(ray)) {
			intersectWorldBox(ray);
			return false;
		}
	}
	
	calculateSelectedWeight();
	return true;
}
	
void ModifyForest::calculateSelectedWeight()
{
	IntersectionContext * ctx = intersection();
	
	selection()->setCenter(ctx->m_hitP, ctx->m_hitN);
	selection()->calculateWeight();
}

float ModifyForest::getNoise() const
{ return m_noiseWeight * (float(rand()%991) / 991.f - .5f); }

float ModifyForest::getNoise2(const float & a, const float & b) const
{ return a + (b - a) * (float(rand()%991) ) / 991.f; }

void ModifyForest::rightUp(GrowOption & option)
{
	if(numActivePlants() < 1) return;
	
	Vector3F useUp = option.m_upDirection;
	PlantSelection::SelectionTyp * arr = activePlants();
	try {
	arr->begin();
	while(!arr->end() ) {
	
		PlantData * plantd = arr->value()->m_reference->index;
		
		if(option.m_alongNormal) {
			useUp = bindNormal(plantd->t2);
		}

		Matrix44F * mat = plantd->t1;
		
		Vector3F vx(mat->M(0, 0), mat->M(0, 1), mat->M(0, 2));
		Vector3F vy(mat->M(1, 0), mat->M(1, 1), mat->M(1, 2));
		Vector3F vz(mat->M(2, 0), mat->M(2, 1), mat->M(2, 2));
		
		const float sx = vx.length();
		const float sy = vy.length();
		const float sz = vz.length();
		
		vx.normalize();
		
		vz = vx.cross(useUp);
		vz.normalize();
		
		vx = useUp.cross(vz);
		vx.normalize();
		
		vx *= sx;
		vy = useUp * sy;
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
	} catch (...) {
		std::cerr<<" ModifyForest rightUp caught something ";
	}
}

void ModifyForest::removeActivePlants()
{
    std::cout<<"\n remove "<<numActivePlants()<<" plants"<<std::endl;
    clearSelected(); 
    selection()->deselect();
    onPlantChanged();
}

void ModifyForest::removeTypedPlants(int x)
{
	if(numPlantExamples() < 1) {
		removeAllPlants();
		return;
	}
	selection()->deselect();
	selectTypedPlants(x);
	std::cout<<"\n select "<<" type"<<x<<" plants"<<std::endl;
	removeActivePlants();
}

bool ModifyForest::sampleGround(EbpGrid * sampler, GrowOption & option)
{
	float gz = 2.f * plantSize(option.m_plantId) + option.m_maxMarginSize;
    std::cout<<"\n flood ground w grid size "<<gz;
    
typedef PrimInd<sdb::Sequence<int>, sdb::VectorArray<cvx::Triangle >, cvx::Triangle > TIntersect;
	TIntersect fintersect(activeGround()->primIndices(), 
	                        &triangles() );
	sampler->fillBox(fintersect, gz * 8.f);
	sampler->subdivideToLevel<TIntersect>(fintersect, 0, 3);
	sampler->insertNodeAtLevel(3);
	sampler->cachePositions();
	std::cout<<"\n flood box    "<<sampler->boundingBox();
	
	for(int i=0;i<20;++i)
		sampler->update();
	
	std::cout<<"\n num sample "<<sampler->numParticles();
	std::cout.flush();
	return true;
}

void ModifyForest::clearPlantOffset(GrowOption & option)
{
	Vector3F pos;
	PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {

		PlantData * plantd = arr->value()->m_reference->index;
		pos = plantd->t1->getTranslation();
		
		GroundBind * bind = plantd->t2;
		
		pos -= bind->m_offset;
		plantd->t1->setTranslation(pos );
		
		bind->m_offset.setZero();
		
		arr->next();
	}
}

void ModifyForest::raiseOffsetAt(const Ray & ray, 
								GrowOption & option)
{
    if(!calculateSelecedWeight(ray)) return;
    
	const Vector3F offsetVec = selectionNormal() * (option.m_strokeMagnitude * plantSize(option.m_plantId) * 10.f);
    std::cout<<"\n offset vec "<<offsetVec;
	Vector3F pos, deltaPos;
    PlantSelection::SelectionTyp * arr = activePlants();
	arr->begin();
	while(!arr->end() ) {
		float wei = arr->value()->m_weight;
		if(wei > 1e-4f) { 
			deltaPos = offsetVec * (wei * (1.f + getNoise() ) );
            
			PlantData * plantd = arr->value()->m_reference->index;
			pos = plantd->t1->getTranslation();
			pos += deltaPos;
			
			plantd->t1->setTranslation(pos);
		
			GroundBind * bind = plantd->t2;
			bind->m_offset += deltaPos;
			
		}
		arr->next();
	}
}

void ModifyForest::getDeltaRotation(Matrix33F & mat,
					const float & weight) const
{ mat.setIdentity(); }

void ModifyForest::getDeltaTranslation(Vector3F & vec,
					const float & weight) const
{ vec.set(0.f, 0.f, 0.f); }

void ModifyForest::getDeltaScaling(Vector3F & vec,
					const float & weight) const
{ vec.set(1.f, 1.f, 1.f); }

void ModifyForest::setManipulatMode(ModifyForest::ManipulateMode x)
{ m_manipulateMode = x; }

ModifyForest::ManipulateMode ModifyForest::manipulateMode() const
{ return m_manipulateMode; }

}