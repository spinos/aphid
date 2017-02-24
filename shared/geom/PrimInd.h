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

#include <geom/ConvexShape.h>
#include <IntersectionContext.h>
#include <geom/ClosestToPointTest.h>

namespace aphid {

template<typename Tind, typename Tsrc, typename Tprim>
class PrimInd : public BoundingBox {
	
	Tind * m_ind;
	const Tsrc * m_src;
	IntersectionContext m_ctx;
    ClosestToPointTestResult m_closestPointTest;
    
public:
	PrimInd(Tind * ind, const Tsrc * src);
	
	bool intersect(const BoundingBox & box);
    bool rayIntersect(const Ray & r);
    bool tetrahedronIntersect(const cvx::Tetrahedron & tet);
    bool closestToPoint(const Vector3F & origin,
                        const float & maxDistance = 1e8f);
	bool select(const Vector3F & center,
				const float & radius);
	bool select(const BoundingBox & bx);
	bool selectedClosestToPoint(const Vector3F & origin,
                        const float & maxDistance = 1e8f);
    Plane closestPlane() const;
    
    const Vector3F & rayIntersectPoint() const;
	const Vector3F & rayIntersectNormal() const;
    const Vector3F & closestToPointPoint() const;
	const Vector3F & closestToPointNormal() const;
    Float2 closestToPointTexcoord() const;
	void getClosestToPointGeomcomp(int & geom, int & comp) const;
    
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
bool PrimInd<Tind, Tsrc, Tprim>::tetrahedronIntersect(const cvx::Tetrahedron & tet)
{
    const Tsrc & rsrc = *m_src;
	m_ind->begin();
	while(!m_ind->end() ) {
		
		const Tprim * t = rsrc[m_ind->key() ];
		
		if(t-> template exactIntersect<cvx::Tetrahedron>(tet) ) {
			return true;
		}
		
		m_ind->next();
	}
    
    return false;
}

template<typename Tind, typename Tsrc, typename Tprim>
bool PrimInd<Tind, Tsrc, Tprim>::closestToPoint(const Vector3F & origin,
                                    const float & maxDistance)
{
    m_closestPointTest.reset(origin, maxDistance);
    const Tsrc & rsrc = *m_src;
	m_ind->begin();
	while(!m_ind->end() ) {
		
		const Tprim * t = rsrc[m_ind->key() ];
		
		t-> template closestToPoint<ClosestToPointTestResult>(&m_closestPointTest);
		
		m_ind->next();
	}
    return m_closestPointTest._hasResult;
    
}

template<typename Tind, typename Tsrc, typename Tprim>
bool PrimInd<Tind, Tsrc, Tprim>::select(const BoundingBox & bx)
{ return true; }

template<typename Tind, typename Tsrc, typename Tprim>
bool PrimInd<Tind, Tsrc, Tprim>::select(const Vector3F & center,
				const float & radius)
{ return true; }

template<typename Tind, typename Tsrc, typename Tprim>
bool PrimInd<Tind, Tsrc, Tprim>::selectedClosestToPoint(const Vector3F & origin,
                        const float & maxDistance)
{
	return closestToPoint(origin, maxDistance);
}

template<typename Tind, typename Tsrc, typename Tprim>
Plane PrimInd<Tind, Tsrc, Tprim>::closestPlane() const
{ return m_closestPointTest.asPlane(); }

template<typename Tind, typename Tsrc, typename Tprim>
const Vector3F & PrimInd<Tind, Tsrc, Tprim>::rayIntersectPoint() const
{ return m_ctx.m_hitP; }

template<typename Tind, typename Tsrc, typename Tprim>
const Vector3F & PrimInd<Tind, Tsrc, Tprim>::rayIntersectNormal() const
{ return m_ctx.m_hitN; }

template<typename Tind, typename Tsrc, typename Tprim>
const Vector3F & PrimInd<Tind, Tsrc, Tprim>::closestToPointPoint() const
{ return m_closestPointTest._hitPoint; }

template<typename Tind, typename Tsrc, typename Tprim>
const Vector3F & PrimInd<Tind, Tsrc, Tprim>::closestToPointNormal() const
{ return m_closestPointTest._hitNormal; }

template<typename Tind, typename Tsrc, typename Tprim>
Float2 PrimInd<Tind, Tsrc, Tprim>::closestToPointTexcoord() const
{ return Float2(0.f, 0.f); }
	
template<typename Tind, typename Tsrc, typename Tprim>
void PrimInd<Tind, Tsrc, Tprim>::getClosestToPointGeomcomp(int & geom, int & comp) const
{}

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