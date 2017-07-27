/*
 *  TestConstraint.h
 *  
 *
 *  Created by jian zhang on 7/28/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_TEST_CONSTRAINT_H
#define APH_TEST_CONSTRAINT_H

#include <pbd/ElasticRodBendAndTwistConstraint.h>

namespace aphid  {

class Vector3F;
class Matrix44F;

namespace pbd {

class TestConstraint : public ElasticRodBendAndTwistConstraint {

public:
	TestConstraint();
	
	void getMaterialFrames(Matrix44F& frmA, Matrix44F& frmB, 
				Vector3F& darboux, Vector3F* correctVs, 
				SimulationContext * model);
	
};

}
}
#endif

