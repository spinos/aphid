/*
 *  VelocityField.h
 *  
 *  stores (u, v, w) at cell center
 *  dimension M x N x P
 *
 *  Created by jian zhang on 1/17/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LBM_VELOCITY_FIELD_H
#define APH_LBM_VELOCITY_FIELD_H

#include "UniformGrid.h"

namespace aphid {

namespace lbm {

class VelocityField : public UniformGrid {

/// marker velocity
	float* m_u[3];
/// marker sum of weight
	float* m_sum[3];
	
public:

	VelocityField();
/// (u, w) <- 0 in d-th dimension with block offset
	void resetBlockVelocity(float* u, float* sum, const int& iblock, const int& d);
/// u <- u + w_i q_i 
/// w <- w + w_i
	void depositeVelocity(const float* pos, const float* vel);
/// u <- u / w
	void finishDepositeVelocity();
/// u_ijk
	void evaluateCellCenterVelocity(float* u, 
				int& i, int& j, int& k) const;
/// U(p) interpolate by centered coord
	void evaluateVelocityAtPosition(float* u, const float* p) const;
/// u is vec3 array
	void extractCellVelocities(float* u) const;
					
protected:

/// all zero
	void clearVelocities();
	void depositeCellCenterVelocity(const int& i, const int& j, const int& k,
				const float* vel);
	
private:
	
	void addVelocity(float* u, float* sum,
			const int& i, const int& j, const int& k,
			const float& xbary, const float& ybary, const float& zbary,
			const float& q,
			const int& d);
	void qdwei(float* q, float* wei, const int& n );
/// d-th component
	void evaluateCellCenterVelocityComponent(float& u, 
				const int& i, const int& j, const int& k,
				const int& d) const;
	
};

}

}

#endif