/*
 *  DiscMesh.h
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_DISC_MESH_H
#define APH_DISC_MESH_H

#include <geom/ATriangleMesh.h>

namespace aphid {
    
class DiscMesh : public ATriangleMesh {
     
public:
    DiscMesh(int nseg = 6);
    virtual ~DiscMesh();
    
protected:
    
private:
    
};

}
#endif