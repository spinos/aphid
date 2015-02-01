#ifndef GJKCONTACTSOLVER_H
#define GJKCONTACTSOLVER_H

/*
 *  GjkContactSolver.h
 *  proof
 *
 *  Created by jian zhang on 1/22/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "Gjk.h"
#ifdef DBG_DRAW
class KdTreeDrawer;
#endif
class GjkContactSolver {
public:
	GjkContactSolver();
#ifdef DBG_DRAW	
	KdTreeDrawer * m_dbgDrawer;
#endif	
	void separateDistance(const PointSet & A, const PointSet & B, ClosestTestContext * result);
	void penetration(const PointSet & A, const PointSet & B, ClosestTestContext * result);
	void rayCast(const PointSet & A, const PointSet & B, ClosestTestContext * result);
	void timeOfImpact(const PointSet & A, const PointSet & B, ContinuousCollisionContext * result);
private:
};
#endif        //  #ifndef GJKCONTACTSOLVER_H
