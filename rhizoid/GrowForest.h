/*
 *  GrowForest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_RHI_GROW_FOREST_H
#define APH_RHI_GROW_FOREST_H

#include "Forest.h"
#include "GrowOption.h"

namespace aphid {

class ForestCell;
class BarycentricCoordinate;

class GrowForest : public Forest {
    
	BarycentricCoordinate * m_ftricoord;
	
public:
	GrowForest();
	virtual ~GrowForest();
	
protected:
	bool growOnGround(GrowOption & option);
	
	bool growAt(const Ray & ray, GrowOption & option);
	bool growAt(const Matrix44F & trans, GrowOption & option);
	
	bool getGroundTriangle(cvx::Triangle & elm, const int & geom,
										const int & comp) const;
	
	bool computeBindContirb(GroundBind & bind,
				cvx::Triangle & elm,
				const Vector3F & pos,
				const int & igeom, const int & icomp);
	 			
private:
	void randomSpaceAt(const Vector3F & pos, 
							const GrowOption & option,
							Matrix44F & space, float & scale);
	
/// when bundleBegin > -1
/// does not collide plant id >= bundleBegin
	bool growSingle(GrowOption & option,
				GroundBind & bind,
				const int & iExample,
				const Matrix44F & tm,
				CollisionContext * collctx);
	void growBundle(GrowOption & option,
				GroundBind & bind,
				const ExampVox * bundle,
				const int & iExample,
				const Matrix44F & tm,
				CollisionContext * collctx);
	void growInCell(ForestCell * cell,
				GrowOption & option,
				GroundBind & bind,
				Matrix44F & tm,
				CollisionContext * collctx);
				
template <typename T>
	bool growOnSample(const T & asmp,
				cvx::Triangle & elm,
				GrowOption & option,
				GroundBind & bind,
				Matrix44F & tm,
				CollisionContext * collctx);
		
	bool processGrow(GrowOption & option,
				GroundBind & bind,
				Matrix44F & tm,
				CollisionContext * collctx);
	
};

template <typename T>
bool GrowForest::growOnSample(const T & asmp,
				cvx::Triangle & elm,
				GrowOption & option,
				GroundBind & bind,
				Matrix44F & tm,
				CollisionContext * collctx)
{
	bind.m_geomComp = asmp.geomcomp;
	
	int igeom, icomp;
	bind.getGeomComp(igeom, icomp);
	
	bool stat = computeBindContirb(bind, elm, asmp.pos, igeom, icomp);
	if(!stat) {
		return stat;
	}

	if(option.m_alongNormal) {
		option.m_upDirection = asmp.nml;
	}
	
	float scaling;
	randomSpaceAt(asmp.pos, option, tm, scaling);
	
	collctx->_radius = plantSize(option.m_plantId) * scaling * .5f;
	collctx->_minDistance = option.m_minMarginSize;
	collctx->_maxDistance = option.m_maxMarginSize;
	collctx->_pos = asmp.pos;
	
	return processGrow(option, bind, tm, collctx);
	
}

}
#endif