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
#include <boost/scoped_array.hpp>
#include <math/linearMath.h>
#include <sdb/WorldGrid2.h>

namespace aphid {

template<typename T>
class DenseVector;

namespace lbm {

class LatticeBlock;

struct LatticeParam {
	
	float _blockSize;
	float _inScale;
	float _outScale;
	int _padding;
	
};

class LatticeManager {

typedef DenseVector<float> QArrayTyp;
typedef sdb::WorldGrid2<LatticeBlock > LatGridTyp;

/// 19 distribution fuction values q_i
/// last one is tmp
	boost::scoped_ptr<QArrayTyp> m_q[20];
/// MAC velocity
	boost::scoped_ptr<QArrayTyp> m_u[3];
/// weight
	boost::scoped_ptr<QArrayTyp> m_sum[3];
/// cell flag
	boost::scoped_array<char > m_flag;
	
/// map to blocks
	LatGridTyp m_grid;
	int m_numBlocks;
	int m_capBlocks;
	LatticeParam m_param;
	
public:
	LatticeManager();
	virtual ~LatticeManager();
	
	void setParam(const LatticeParam& param);

/// 16 empty blocks 
	void resetLattice();
/// transfer np particles into grid p is position v is velocity
	void injectParticles(const float* p,
					const float* v,
					const int& np);
	void finishInjectingParticles();
	
	LatGridTyp& grid();
	
	QArrayTyp& q_i(const int& i);
	
	void initialCondition();
/// streaming and collision for all blocks
	void simulationStep();
/// vel	and pos in vec3[np]	
	void modifyParticleVelocities(float* vel,
					const float* pos,
					const int& np);
	
protected:

private:
	void extendArrays();
	void resetBlock(LatticeBlock* blk, const float& cx, const float& cy, const float& cz);
	
};

}

}

#endif
