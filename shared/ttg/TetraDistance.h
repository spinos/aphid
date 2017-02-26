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

#include <math/Plane.h>
#include <geom/ConvexShape.h>
#include <math/miscfuncs.h>

namespace aphid {

namespace cvx {
class Tetrahedron;
}

namespace ttg {

class TetraDistance : public cvx::Tetrahedron {
    
    float m_d[4];
    bool m_valid[4];
    
public:
    TetraDistance();
    virtual ~TetraDistance();
    
    template<typename Tf>
    void compute(Tf * intersectF, const float & maxDistance,
                const float & offset)
    {
        for(int i=0;i<4;++i) {
			m_valid[i] = false;
			if(!intersectF->selectedClosestToPoint(P(i), maxDistance) ) {
                continue;
            }

            m_valid[i] = true;
            Vector3F dv = intersectF->closestToPointPoint() - P(i);
			float ldv = dv.length();
           if(ldv > 1e-3f) {
                dv /= ldv;
                const float dvdotn = intersectF->closestToPointNormal().dot(dv);
				if(dvdotn > 0.f) {
/// back side					
					if(dvdotn < .2f) {
/// no effect
						m_valid[i] = false;
						
					} else if(dvdotn > .8f) {
/// inside
						ldv = -ldv;
					}
				} 
			}
            m_d[i] = ldv  - offset;
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
