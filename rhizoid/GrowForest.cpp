/*
 *  GrowForest.cpp
 *  proxyPaint
 *
 *	sample based growth
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "GrowForest.h"
#include "ExampVox.h"
#include "ForestGrid.h"
#include "ForestCell.h"
#include <geom/ATriangleMesh.h>
#include <ctime>

namespace aphid {

GrowForest::GrowForest()
{
	m_ftricoord = new BarycentricCoordinate;
	std::time_t tim(NULL);
	srand((int)tim);
}

GrowForest::~GrowForest() 
{
	delete m_ftricoord;
}

bool GrowForest::growOnGround(GrowOption & option)
{	
	ForestGrid * fg = grid();
	if(!fg) {
		std::cout<<"\n GrowForest has no grid";
		return false;
	}
	
    if(fg->numVisibleSamples() < 1) {
		std::cout<<"\n GrowForest has no visible samples";
		return false;
	}
	
	GroundBind bind;
	Matrix44F tm;
	Vector3F pog;
	CollisionContext collctx;
	collctx._minIndex = -1;
	
	try {
	
/// for each active cell
	fg->activeCellBegin();
	while(!fg->activeCellEnd() ) {

		ForestCell * fcell = fg->activeCellValue(); 
		if(fcell->numVisibleSamples() > 0) {
			growInCell(fcell, option, bind, tm, &collctx);
		}
		
		fg->activeCellNext();
		
	}
	} catch (const char * ex) {
	    std::cerr<<"GrowForest growOnGround caught: "<<ex;
	} catch (...) {
	    std::cerr<<"GrowForest growOnGround caught something";
	}
	
	return true;
}

void GrowForest::growBundle(GrowOption & option,
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

bool GrowForest::growSingle(GrowOption & option,
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

bool GrowForest::growAt(const Ray & ray, GrowOption & option)
{
	if(!intersectGround(ray) ) {
		intersectWorldBox(ray);
		return false;
	}
	
	IntersectionContext * ctx = intersection();
	
	if(option.m_multiGrow) {
		bool stat = selectGroundSamples(ray, SelectionContext::Replace);
		if(!stat) {
			return false;
		}
		onSampleChanged();
		
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
	
	m_ftricoord->create(t->P(0), t->P(1), t->P(2) );
    
	GroundBind bind;
	bind.setGeomComp(t->ind0(), t->ind1() );
	
	m_ftricoord->project(ctx->m_hitP);
	m_ftricoord->compute();
	if(!m_ftricoord->insideTriangle()) {
		std::cout<<"\n out of triangle pnt "<<ctx->m_hitP;
		return false;
	}
	
	bind.m_w0 = m_ftricoord->getV(0);
	bind.m_w1 = m_ftricoord->getV(1);
	bind.m_w2 = m_ftricoord->getV(2);
		
	Matrix44F tm;
	float scale;
	randomSpaceAt(ctx->m_hitP, option, tm, scale); 
	
	CollisionContext collctx;
	collctx._minIndex = -1;
	collctx._radius = plantSize(option.m_plantId) * scale * .5f;
	collctx._minDistance = option.m_minMarginSize;
	collctx._maxDistance = option.m_maxMarginSize;
	collctx._pos = ctx->m_hitP;
	
	const ExampVox * v = plantExample(option.m_plantId);
	if(v->numExamples() > 1) {
		growBundle(option, bind, v, option.m_plantId, tm, &collctx);			
	
	} else {
		growSingle(option, bind, option.m_plantId, tm, &collctx);
				
	}
		
    return true;
}

bool GrowForest::growAt(const Matrix44F & trans, GrowOption & option)
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

void GrowForest::randomSpaceAt(const Vector3F & pos, const GrowOption & option,
							Matrix44F & space, float & scale)
{
	*space.m(3, 0) =  pos.x;
	*space.m(3, 1) =  pos.y;
	*space.m(3, 2) =  pos.z;
	
	Vector3F up = option.m_upDirection;
	
	Vector3F side(1.f, 0.f, 0.f);
	if(up.x > 0.9f || up.x < -0.9f) {
		side = Vector3F::YAxis;
	}
		
	Vector3F front = side.cross(up);
	front.normalize();
	
	side = up.cross(front);
	side.normalize();
	
	scale = option.m_minScale + RandomF01() * (option.m_maxScale - option.m_minScale);
	
	Vector3F vside(side.x, side.y, side.z);
	Vector3F vaxis(up.x, up.y, up.z);
	float ang = (RandomF01() - 0.5f) * option.m_rotateNoise * 2.f * 3.14f;
	
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

void GrowForest::growInCell(ForestCell * cell,
				GrowOption & option,
				GroundBind & bind,
				Matrix44F & tm,
				CollisionContext * collctx)
{
	const sdb::SampleCache * sc = cell->sampleCacheAtLevel(sampleLevel() );
	cvx::Triangle ielm;
	const int & n = cell->numVisibleSamples();
	for(int i=0;i<n;++i) {
/// i-th visible sample	
		const sdb::SampleCache::ASample & asmp = sc->getASample(cell->visibleSampleIndices()[i]);
		
		growOnSample<sdb::SampleCache::ASample>(asmp, ielm, option, bind, tm, collctx);
	}
	
}

bool GrowForest::getGroundTriangle(cvx::Triangle & elm, const int & geom,
										const int & comp) const
{
	const ATriangleMesh * mesh = getGroundMesh(geom);
	if(!mesh) {
		return false;
	}
	
	if(mesh->numTriangles() <= comp) {
		return false;
	}
	
	mesh->dumpComponent<cvx::Triangle>(elm, comp);
	
	return true;
}

bool GrowForest::computeBindContirb(GroundBind & bind,
				cvx::Triangle & elm,
				const Vector3F & pos,
				const int & igeom, const int & icomp)
{
	bool stat = getGroundTriangle(elm, igeom, icomp);
	if(!stat) {
		return stat;
	}
	
	m_ftricoord->create(elm.P(0), elm.P(1), elm.P(2) );
    m_ftricoord->project(pos);
	m_ftricoord->compute();
	
	stat = m_ftricoord->insideTriangle();
	if(!stat) {
		std::cout<<"\n out of triangle pnt "<<pos;
		return stat;
	}
	
	bind.m_w0 = m_ftricoord->getV(0);
	bind.m_w1 = m_ftricoord->getV(1);
	bind.m_w2 = m_ftricoord->getV(2);
	return stat;
}

bool GrowForest::processGrow(GrowOption & option,
				GroundBind & bind,
				Matrix44F & tm,
				CollisionContext * collctx)
{
	const ExampVox * v = plantExample(option.m_plantId);
	if(v->numExamples() > 1) {
		growBundle(option, bind, v, option.m_plantId, tm, collctx);			
		return true;
	} 
	
	return growSingle(option, bind, option.m_plantId, tm, collctx);
	
}

}