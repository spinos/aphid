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
    bool m_valid[4];
    float m_maxDistance;
    
public:
    TetraDistance(const cvx::Tetrahedron & tet);
    virtual ~TetraDistance();
    
    template<typename Tf>
    void compute(Tf * intersectF)
    {
        for(int i=0;i<4;++i) {
            if(!intersectF->closestToPoint(m_p[i]) ) {
                continue;
            }
            m_valid[i] = true;
            Plane pl = intersectF->closestPlane();
            m_d[i] = pl.distanceTo(m_p[i]);
            Vector3F dv = intersectF->closestToPointPoint() - m_p[i];
            const float ldv = dv.length();
            if(ldv > m_maxDistance) {
                m_valid[i] = false;
            } else if(ldv > 1e-2f) {
                dv /= ldv;
                const float dvdotn = Absolute<float>(intersectF->closestToPointNormal().dot(dv) );
                if(dvdotn < .3f) {
                    m_valid[i] = false;
                } else if(dvdotn < .8f) {
                    m_d[i] = ldv;
                }
            }
            
        }
    }
    
    void compute(const Plane & pl);
    
    const float * result() const;
    
    const bool * isValid() const;
    
protected:

private:
};

}

}
#endif
