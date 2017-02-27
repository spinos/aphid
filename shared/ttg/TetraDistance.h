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
    
	Vector3F m_pvs[4];
    float m_d[4];
    bool m_valid[4];
	int m_indices[4];
	Vector3F m_snapPos;
	int m_snapInd;
    
public:
    TetraDistance();
    virtual ~TetraDistance();
	
	void setDistance(const float & x, int i);
	void setIndices(const int * x);
	
	const int & snapInd() const;
	const Vector3F & snapPos() const;
    
    template<typename Tf>
    void compute(Tf * intersectF, const float & rDistance,
                const float & offset,
				const float & snapThreshold)
    {
		m_snapInd = -1;
        for(int i=0;i<4;++i) {
/// reset
			m_valid[i] = false;
			
/// already on front or inside			
			if(m_d[i] <= 0.f) {
				continue;
			}
			
/// to far away
			if(!intersectF->selectedClosestToPoint(P(i), rDistance * 3.f + offset) ) {
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
				} else {
					if(dvdotn > -.2f) {
						m_valid[i] = false;
					}
				}
			} else {
				ldv = 0.f;
			}
			
			if(!m_valid[i]) {
				continue;
			}
			
			ldv -= offset;
			
			m_pvs[i] = intersectF->closestToPointPoint() 
						+ intersectF->closestToPointNormal() * offset;
						
			if(m_d[i] > ldv) {
				m_d[i] = ldv;
			}
        }
		
/// same for all tetra
		float minD = snapThreshold;
/// the one closest to front
		for(int i=0;i<4;++i) {
			if(!m_valid[i]) {
				continue;
			}
			float d = Absolute<float>(m_d[i]);
			if(minD > d) {
				minD = d;
				m_snapInd = i;
			}
		}
		
		if(m_snapInd > -1) {
/// move to front
			m_snapPos = m_pvs[m_snapInd];//P(m_snapInd);//
			m_d[m_snapInd] = 0.f;
			m_snapInd = m_indices[m_snapInd];
			
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
