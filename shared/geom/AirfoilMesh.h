/*
 *  AirfoilMesh.h
 *
 *  Created by jian zhang on 12/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_GEOM_AIRFOIL_MESH_H
#define APH_GEOM_AIRFOIL_MESH_H

#include <geom/Airfoil.h>
#include <geom/ATriangleMesh.h>

namespace aphid {

class AirfoilMesh : public Airfoil, public ATriangleMesh {

public:
	AirfoilMesh(const float & c,
			const float & m,
			const float & p,
			const float & t);
	virtual ~AirfoilMesh();
	
/// gx n grid along x, even
//. gy n grid along y
	void tessellate(const int & gx = 20, const int & gy = 2);
	
	void flipAlongChord();
	
protected:

private:

};

}

#endif
