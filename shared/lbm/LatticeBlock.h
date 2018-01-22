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
/// density
	float* m_rho;
/// fluid, bounduary condition, obstacle
	char* m_flag;
	static float* RestQ[19];
	
public:

	LatticeBlock(sdb::Entity * parent = NULL);
	
/// q_i <- w_i with block offset
	void resetQi(float* q, const int& iblock, const int& i);
	void resetDensity(float* rho, const int& iblock);
	void resetFlag(char* fg, const int& iblock);
	
/// find incomming cells
/// initial condition prescribed in terms of fluid velocity
/// f_i^0 <- f_i^eq(rho_i(x), u_0(x))
/// zero velocity f_i^0 <- w_i
/// flag_i <- 0
	void initialCondition();
	void simulationStep();
/// cell center to MAC
	void updateVelocityDensity();
	
	void extractCellDensities(float* dst);
	void evaluateVelocityDensityAtPosition(float* u, float& rho, const float* p);
	void modifyParticleVelocities(float* vel, const float* pos,
					const int& np, const float& scaling);
	
	static void BuildRestQ();
	
protected:
	virtual void limitSpeed(float& x) const;
	
private:
/// propagate i-th distribution function values f_i to the next lattice site 
/// in the direction of its assigned lattice vector c_i
	void streaming(const int& i);
	void collision();
	void boundaryCondition(const float& bcRho);
/// from rank zbeing to zend-1
	void rankedInitialCondition(const int& zbegin, const int& zend);
/// from particle ibegin to iend-1
	void blockModifyParticleVelocities(float* vel, const float* pos, const float& scaling,
					const int& ibegin, const int& iend);
	
};

}

}
#endif
