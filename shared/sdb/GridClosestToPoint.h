/*
 *  GridClosestToPoint.h
 *
 *  T as grid type, Tc as cell type, Tn as node type
 *  closest to node in selected cells 
 *
 *  Created by jian zhang on 2/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_GRID_CLOSEST_TO_POINT_H
#define APH_SDB_GRID_CLOSEST_TO_POINT_H

#include "GridSelection.h"
#include <geom/ClosestToPointTest.h>

namespace aphid {

namespace sdb {

template<typename T, typename Tc, typename Tn>
class GridClosestToPoint : public GridSelection<T, Tc> {

	ClosestToPointTestResult m_ctx;
	
typedef GridSelection<T, Tc> SelectionTyp;
	
public:
	GridClosestToPoint(T * grid);
	
	bool selectedClosestToPoint(const Vector3F & origin,
                        const float & maxDistance = 1e8f);
	
	const Vector3F & closestToPointPoint() const;
	const Vector3F & closestToPointNormal() const;
    
protected:

private:
};

template<typename T, typename Tc, typename Tn>
GridClosestToPoint<T, Tc, Tn>::GridClosestToPoint(T * grid) :
GridSelection<T, Tc>(grid)
{}

template<typename T, typename Tc, typename Tn>
bool GridClosestToPoint<T, Tc, Tn>::selectedClosestToPoint(const Vector3F & origin,
                        const float & maxDistance)
{
	m_ctx.reset(origin, maxDistance);
	
	const int nc = SelectionTyp::numActiveCells();
	for(int i=0;i<nc;++i) {
		Tc * ac = SelectionTyp::activeCell(i);
		ac-> template closestToPoint<ClosestToPointTestResult>(&m_ctx);
		
		if(m_ctx.closeEnough() ) {
			break;
		}
		
	}
	return m_ctx._hasResult;
    
}

template<typename T, typename Tc, typename Tn>
const Vector3F & GridClosestToPoint<T, Tc, Tn>::closestToPointPoint() const
{ return m_ctx._hitPoint; }

template<typename T, typename Tc, typename Tn>
const Vector3F & GridClosestToPoint<T, Tc, Tn>::closestToPointNormal() const
{ return m_ctx._hitNormal; }

}

}
#endif
