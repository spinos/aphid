/*
 *  UniformGrid.h
 *  
 *
 *  Created by jian zhang on 1/19/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LBM_UNIFORM_GRID_H
#define APH_LBM_UNIFORM_GRID_H

namespace aphid {

namespace lbm {

class UniformGrid {

/// lower corner
	float m_corner[3];
	
public:
	UniformGrid();

	void setCorner(const float& x, const float& y, const float& z);
	
	bool isPointOutsideBound(const float* pos) const;

/// in cell coord weight to lower corner	
	void getCellCoordWeight(int& i, int& j, int& k,
				float& barx, float& bary, float& barz,
				const float* u) const;
	int getCellInd(const float* u) const;
	int getCellZCoord(const float* u) const;

/// p is vec3 array
	void extractCellCenters(float* p) const;
	
/// MNk + Mj + i
	static int CellInd(const int& i, const int& j, const int& k);
/// k <- ind / MN
/// j <- (ind - kMN) / M
/// i <- ind - kMN - jM
	static void CellCoord(int& i, int& j, int& k, const int& ind);
	
	static bool IsCellCoordValid(const int& i, const int& j, const int& k);
	
/// c[2][4] is 8 corners
/// 000 100 010 110
/// 001 101 011 111
/// http://www.cs-fundamentals.com/c-programming/arrays-in-c.php
	static void TrilinearInterpolation(float& u, float c[][4], 
				const float& barx, const float& bary, const float& barz );
				
	static int BlockDim[3];
/// MNP
	static int BlockLength;
/// h
	static float CellSize;
/// h / 2
	static float HalfCellSize;
/// 1 / h
	static float OneOverH;
				
protected:
/// in d-th dimension
	static void CenteredCoordWeight(int& i, float& bary, const int& d);
/// divide ind to 8 parts in z direction	
	static int ZInd8Begins[9];
/// divide rank to 8 parts in z direction
	static int ZRank8Begins[9];
	
private:
	
};

}

}

#endif