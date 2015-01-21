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

class GjkContactSolver {
public:
	GjkContactSolver();
	
	char pairContacted(const PointSet & A, const PointSet & B);
private:
};
#endif        //  #ifndef GJKCONTACTSOLVER_H
