/*
 *  PointDistance.h
 *  
 *
 *  Created by zhang on 17-2-4.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_POINT_DISTANCE_H
#define APH_TTG_POINT_DISTANCE_H

#include <math/Vector3F.h>
#include <math/miscfuncs.h>

namespace aphid {

namespace ttg {

class PointDistance {
    
	Vector3F m_pos;
	float m_d;
    bool m_valid;
    
public:
    PointDistance();
    virtual ~PointDistance();
	
	void setPos(const Vector3F & v);
	const Vector3F & pos() const;
    
    template<typename Tf>
    void compute(Tf * intersectF, const float & rDistance,
                const float & snapThreshold);
    
    const float & result() const;
    const bool & isValid() const;
    
protected:

private:
};

template<typename Tf>
void PointDistance::compute(Tf * intersectF, const float & rDistance,
                const float & snapThreshold)
    {
/// reset
		m_valid = false;
			
/// to far away
		if(!intersectF->selectedClosestToPoint(m_pos, rDistance) ) {
			return;
		}
			
		m_valid = true;
			
		Vector3F dv = intersectF->closestToPointPoint() - m_pos;
		float ldv = dv.length();
		if(ldv > 1e-2f) {
			dv /= ldv;
			const float dvdotn = intersectF->closestToPointNormal().dot(dv);
			if(dvdotn > 0.f) {
/// back side					
				if(dvdotn < .2f) {
/// no effect
					m_valid = false;
						
				} else if(dvdotn > .5f) {
/// inside
					ldv = -ldv;
				}
			} else {
				if(dvdotn > -.2f) {
					m_valid = false;
				}
			}
		} else {
			ldv = 0.f;
		}
			
		if(!m_valid) {
			return;
		}
			
		m_d = ldv;
		
		if(Absolute<float>(m_d) < snapThreshold) {
			m_d = 0.f;
			m_pos = intersectF->closestToPointPoint();
		}
		
    }

}

}
#endif
