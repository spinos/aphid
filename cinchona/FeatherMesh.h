/*
 *  FeatherMesh.h
 *  
 *  airfoil flip x facing to +y
 *
 *  Created by jian zhang on 1/3/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef FEATHER_MESH_H
#define FEATHER_MESH_H

#include <geom/AirfoilMesh.h>

class FeatherMesh : public aphid::AirfoilMesh {

	aphid::Vector3F * m_leadingEdgeVertices;
	int m_numLeadingEdgeVertices;
	
public:
	FeatherMesh(const float & c,
			const float & m,
			const float & p,
			const float & t);
	virtual ~FeatherMesh();
	
/// tessellate, flip, rotate
	void create(const int & gx,
				const int & gy);
	
	const int & numLeadingEdgeVertices() const;
	const aphid::Vector3F * leadingEdgeVertices() const;
	
protected:

private:
};
#endif