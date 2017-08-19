/*
 *  BillboardMesh.h
 *
 *  grid mesh with width, height, nu = 1 
 *  centered at (0, height/2) along xy plane facing +z
 *
 *  Created by jian zhang on 8/9/17.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#ifndef APH_BILLBOARD_MESH_H
#define APH_BILLBOARD_MESH_H

#include "GridMesh.h"

namespace aphid {
    
class BillboardMesh : public GridMesh {
     
public:
	BillboardMesh();
    virtual ~BillboardMesh();
	
	virtual void setBillboardSize(float w, float h, int nu, int addNv);
	
protected:
    
private:
    
};

}
#endif
