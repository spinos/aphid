/*
 *  LatticeManager.h
 *  
 *  D3Q19 scheme for lattice-Boltzmann method
 *
 *  Created by jian zhang on 1/15/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LBM_LATTICE_MANAGER_H
#define APH_LBM_LATTICE_MANAGER_H

#include <boost/scoped_ptr.hpp>
#include <math/linearMath.h>
#include <sdb/WorldGrid2.h>

namespace aphid {

namespace lbm {

class LatticeBlock;

struct LatticeParam {
	
	float _blockSize;
	
};

class LatticeManager {

typedef DenseVector<float> QArrayTyp;
typedef sdb::WorldGrid2<LatticeBlock > LatGridTyp;

/// 19 discrete velocities q
	boost::scoped_ptr<QArrayTyp> m_q[19];
/// map to blocks
	LatGridTyp m_grid;
	int m_numBlocks;
	int m_capBlocks;
	
public:
	LatticeManager();
	virtual ~LatticeManager();

/// 16 empty blocks 
	void resetLattice(const LatticeParam& param);
/// transfer np particles into grid p is position v is velocity
	void injectParticles(const float* p,
					const float* v,
					const int& np);
	
	LatGridTyp& grid();
	
protected:

private:
	void extendQ();
	void initializeBlockQ(const int& begin);
	void addVelocity(const int& u, const int& v, const int& w,
				const float& bu, const float& bv, const float& bw,
				const float* vel,
				const int& begin);
	
};

}

}

#endif
