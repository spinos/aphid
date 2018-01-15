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

namespace aphid {

namespace lbm {

class LatticeBlock : public sdb::Entity {

/// lower corner
	float m_corner[3];
/// where q begins
	int m_qoffset;
	
public:

	LatticeBlock(sdb::Entity * parent = NULL);
	
	void setCorner(const float& x, const float& y, const float& z);
	void setQOffset(const int& x);
	
	const int& qOffset() const;
	
/// |       |       |
/// |   i  x|   i1  |
/// |       |       |
/// ind to node and barycentric coordinate in d-th dimension
/// if i = -1, outside lower bound
	void calcNodeCoord(int& i, float& bary, const float& u, const int& d) const;
	
/// q_i <- w_i
	static void InitializeQ(float* q, const int& i);
	static void AddQ(const int& u, const int& v, const int& w,
				const float* vel,
				float* q, const int& i);
	static int NodeInd(const int& i, const int& j, const int& k);
	static bool IsNodeIndOutOfBound(const int& i, const int& j, const int& k);

/// h
	static float NodeSize;
/// h / 2
	static float HalfNodeSize;
/// 1 / h
	static float OneOverH;
/// 16 x 16 x 16
	static int BlockLength;
	
protected:

private:
	
};

}

}
#endif
