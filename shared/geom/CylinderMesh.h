/*
 *  CylinderMesh.h
 *
 *  a cylinder 
 *  originate at zero growing up with radius
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_CYLINDER_MESH_H
#define APH_CYLINDER_MESH_H

#include <geom/ATriangleMesh.h>

namespace aphid {
    
class CylinderMesh : public ATriangleMesh {

/// num cells u and v
	int m_nu, m_nv;
	float m_radius, m_circum, m_height;
		
public:
	CylinderMesh();
    CylinderMesh(int nu, int nv, float radius, float dv);
    virtual ~CylinderMesh();
	
	virtual void createCylinder(int nu, int nv, float radius, float height);
	
    const int& nu() const;
	const int& nv() const;
	const float& radius() const;
/// nu seg_u r
	const float& circumference() const;
	const float& height() const;
/// nu seg_u r / h
	float circumferenceHeightRatio() const;
		
protected:
/// nv+1 segment heights provided
    void createCylinder1(int nu, int nv, float radius, float height,
		const float* heightSegs);
	
private:
    void addOddCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& i1, const int& j1,
				const int& nu,
				const float& du, const float* heightSegs,
				Vector2F* fvps);
	void addEvenCell(unsigned* ind, int& tri,
				const int& i, const int& j,
				const int& i1, const int& j1,
				const int& nu,
				const float& du, const float* heightSegs,
				Vector2F* fvps);
/// (x,y) to uv with v fill [0,1]
	void projectTexcoord();
	
};

}
#endif
