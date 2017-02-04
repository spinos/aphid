/*
 *  TetraDistance.h
 *  
 *
 *  Created by zhang on 17-2-4.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_TETRA_DISTANCE_H
#define APH_TTG_TETRA_DISTANCE_H

#include <ConvexShape.h>
#include <math/Plane.h>

namespace aphid {

class Plane;

namespace ttg {

class TetraDistance {
    
    Vector3F m_p[4];
    float m_d[4];
    
public:
    TetraDistance(const cvx::Tetrahedron & tet);
    virtual ~TetraDistance();
    
    template<typename Tf>
    void compute(Tf * intersectF)
    {
        for(int i=0;i<4;++i) {
            intersectF->closestToPoint(m_p[i]);
            Plane pl = intersectF->closestPlane();
            m_d[i] = pl.distanceTo(m_p[i]);
        }
    }
    
    void compute(const Plane & pl);
    
    const float * result() const;
    
protected:

private:
};

}

}
#endif
