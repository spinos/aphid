/*
 *  PrimInd.h
 *  
 *  intersection util
 *
 *  Created by zhang on 17-1-31.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GEOM_PRIM_IND_H
#define APH_GEOM_PRIM_IND_H

#include <math/BoundingBox.h>
#include <IntersectionContext.h>

namespace aphid {

template<typename Tind, typename Tsrc, typename Tprim>
class PrimInd : public BoundingBox {
	
	Tind * m_ind;
	const Tsrc * m_src;
	IntersectionContext m_ctx;
    
public:
	PrimInd(Tind * ind, const Tsrc * src);
	
	bool intersect(const BoundingBox & box);
    bool rayIntersect(const Ray & r);
    
    const Vector3F & rayIntersectPoint() const;
	const Vector3F & rayIntersectNormal() const;
    
    void getAggregatedPositionNormal(Vector3F & resultP,
                        Vector3F& resultN);
                        
};

template<typename Tind, typename Tsrc, typename Tprim>
PrimInd<Tind, Tsrc, Tprim>::PrimInd(Tind * ind, const Tsrc * src)
{
    m_ind = ind;
    m_src = src;
    const Tsrc & rsrc = *src;
    m_ind->begin();
    while(!m_ind->end() ) {
        
        const Tprim * t = rsrc[m_ind->key() ];
        expandBy(t->calculateBBox() );
        
        m_ind->next();
    }
}

template<typename Tind, typename Tsrc, typename Tprim>
bool PrimInd<Tind, Tsrc, Tprim>::intersect(const BoundingBox & box)
{
	if(!box.intersect(*this) ) return false;
	
	const Tsrc & rsrc = *m_src;
	m_ind->begin();
	while(!m_ind->end() ) {
		
		const Tprim * t = rsrc[m_ind->key() ];
		if(box.intersect(t->calculateBBox() ) ) {
		if(t-> template exactIntersect<BoundingBox >(box) )
			return true;
		}
		
		m_ind->next();
	}
	return false;
}

template<typename Tind, typename Tsrc, typename Tprim>
bool PrimInd<Tind, Tsrc, Tprim>::rayIntersect(const Ray & r)
{
    m_ctx.reset(r);
    
    const Tsrc & rsrc = *m_src;
	m_ind->begin();
	while(!m_ind->end() ) {
		
		const Tprim * t = rsrc[m_ind->key() ];
		
		if(t->rayIntersect(m_ctx.m_ray, &m_ctx.m_tmin, &m_ctx.m_tmax) ) {
			m_ctx.m_hitP = m_ctx.m_ray.travel(m_ctx.m_tmin);
			m_ctx.m_hitN = t->calculateNormal();
/// shorten ray
			m_ctx.m_ray.m_tmax = m_ctx.m_tmin;
			m_ctx.m_success = 1;
/// idx of source
			m_ctx.m_componentIdx = m_ind->key();
		}
		
		m_ind->next();
	}
    
    return m_ctx.m_success;
}

template<typename Tind, typename Tsrc, typename Tprim>
const Vector3F & PrimInd<Tind, Tsrc, Tprim>::rayIntersectPoint() const
{ return m_ctx.m_hitP; }

template<typename Tind, typename Tsrc, typename Tprim>
const Vector3F & PrimInd<Tind, Tsrc, Tprim>::rayIntersectNormal() const
{ return m_ctx.m_hitN; }

template<typename Tind, typename Tsrc, typename Tprim>
void PrimInd<Tind, Tsrc, Tprim>::getAggregatedPositionNormal(Vector3F & resultP,
                        Vector3F& resultN)
{
    resultP.set(0.f, 0.f, 0.f);
    resultN.set(0.f, 0.f, 0.f);
    
    int c = 0;
    const Tsrc & rsrc = *m_src;
	m_ind->begin();
	while(!m_ind->end() ) {
		
		const Tprim * t = rsrc[m_ind->key() ];
		
        resultP += t->center();
        resultN += t->calculateNormal();
		c++;
		
		m_ind->next();
	}
    
    resultP /= (float)c;
    resultN.normalize();
    
}

}

#endif