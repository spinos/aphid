/*
 *  PlanarTexcoordProjector.h
 *  
 *
 *  Created by jian zhang on 9/5/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PLANAR_TEXCOORD_PROJECTOR_H
#define APH_PLANAR_TEXCOORD_PROJECTOR_H

namespace aphid {

class ATriangleMesh;
class BoundingBox;

class PlanarTexcoordProjector {
	
public:
	enum TexcoordOrigin {
		tLeftBottom = 0,
		tCenteredZero,
		tCenteredBox,
	};
	
	PlanarTexcoordProjector();
	
	void setTexcoordOrigin(TexcoordOrigin x);
	
/// x-y plane	
	virtual void projectTexcoord(ATriangleMesh* msh,
					BoundingBox& bbx) const;
	
private:
	TexcoordOrigin m_texcori;
	
};

}

#endif
