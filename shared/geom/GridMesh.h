/*
 *  GridMesh.h
 *
 *  mesh forms a uniform grid 
 *  originate at zero along xy plane facing +z
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_GRID_MESH_H
#define APH_GRID_MESH_H

#include <geom/ATriangleMesh.h>

namespace aphid {
    
class GridMesh : public ATriangleMesh {

/// num cells u and v
	int m_nu, m_nv;
	float m_du, m_dv;
		
public:
	GridMesh();
    GridMesh(int nu, int nv, float du, float dv);
    virtual ~GridMesh();
	
    const int& nu() const;
	const int& nv() const;
	float width() const;
	float height() const;
/// w / h
	float widthHeightRatio() const;
		
protected:
    void createGrid(int nu, int nv, float du, float dv);

private:
    void addOddCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& nu1);
	void addEvenCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& nu1);
/// (x,y) to uv with v fill [0,1]
	void projectTexcoord();
	
};

}
#endif
