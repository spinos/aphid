/*
 *  LatticeBlock.h
 *  
 *  16 x 16 x 16 nodes
 *
 *  Created by jian zhang on 1/16/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LBM_LATTICE_BLOCK_H
#define APH_LBM_LATTICE_BLOCK_H

#include <sdb/Entity.h>
#include "MACVelocityField.h"

namespace aphid {

template<typename T>
class DenseVector;

namespace lbm {

class LatticeBlock : public sdb::Entity, public MACVelocityField {

/// 19 distribution functions values f_i and 1 tmp buffer
	float* m_q[20];
/// fluid, bounduary condition, obstacle
	char* m_flag;
	
public:

	LatticeBlock(sdb::Entity * parent = NULL);
	
/// q_i <- w_i with block offset
	void resetQi(float* q, const int& iblock, const int& i);
	void resetFlag(char* fg, const int& iblock);
	
	void simulationStep();
/// cell center to MAC
	void updateVelocity();
	
protected:

private:
/// propagate i-th distribution function values f_i to the next lattice site 
/// in the direction of its assigned lattice vector c_i
	void streaming(const int& i);
	void collision();
/// find incomming cells
/// set initial velocities
	void initialCondition();
	void boundaryCondition(const float& bcRho);
	
};

}

}
#endif
