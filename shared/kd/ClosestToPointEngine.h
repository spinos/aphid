/*
 *  ClosestToPointEngine.h
 *  
 *
 *  Created by jian zhang on 1/13/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_KD_CLOSEST_TO_POINT_ENGINE_H
#define APH_KD_CLOSEST_TO_POINT_ENGINE_H

#include <kd/SelectEngine.h>
#include <math/Quantization.h>

namespace aphid {

template<typename T, typename Tn>
class ClosestToPointEngine : public SelectEngine<T, Tn> {

	ClosestToPointTestResult m_ctx;
	
public:

	ClosestToPointEngine(KdNTree<T, Tn > * tree);
    
    bool closestToPoint(const Vector3F & origin,
                        const float & maxDistance = 1e8f);
						
	bool selectedClosestToPoint(const Vector3F & origin,
                        const float & maxDistance = 1e8f);
	
	void getGeomCompContribute(int & igeom,
					int & icomp,
					float * contrib) const;

    const Vector3F & closestToPointPoint() const;
	const Vector3F & closestToPointNormal() const;
    Float2 closestToPointTexcoord() const;
	void getClosestToPointGeomcomp(int & geom, int & comp) const;
    Plane closestPlane() const;
	Vector3F closestToPointColor() const;
	int closestToPointColorCompressed() const;
	
};

template<typename T, typename Tn>
ClosestToPointEngine<T, Tn>::ClosestToPointEngine(KdNTree<T, Tn > * tree) :
SelectEngine<T, Tn>(tree)
{}

template<typename T, typename Tn>
bool ClosestToPointEngine<T, Tn>::closestToPoint(const Vector3F & origin,
                                    const float & maxDistance)
{
    m_ctx.reset(origin, maxDistance);
    KdEngine::closestToPoint<T, Tn>(SelectEngine<T, Tn>::tree(), &m_ctx);
    return m_ctx._hasResult;
    
}

template<typename T, typename Tn>
bool ClosestToPointEngine<T, Tn>::selectedClosestToPoint(const Vector3F & origin,
                        const float & maxDistance)
{
	m_ctx.reset(origin, maxDistance);
	
	sdb::Sequence<int> * prims = SelectEngine<T, Tn>::primIndices();
	const sdb::VectorArray<T> & src = SelectEngine<T, Tn>::source();
	
	prims->begin();
	while(!prims->end() ) {
	
		m_ctx._currentSourceInd = prims->key();
		const T * ts = src[m_ctx._currentSourceInd];
		ts-> template closestToPoint<ClosestToPointTestResult>(&m_ctx);
		
		if(m_ctx.closeEnough() ) {
			break;
		}
		
		prims->next();
	}
	return m_ctx._hasResult;
    
}

template<typename T, typename Tn>
void ClosestToPointEngine<T, Tn>::getGeomCompContribute(int & igeom,
					int & icomp,
					float * contrib) const
{
	igeom = m_ctx._igeometry;
	icomp = m_ctx._icomponent;
/// up to 4 for tetrahedron
	contrib[0] = m_ctx._contributes[0];
	contrib[1] = m_ctx._contributes[1];
	contrib[2] = m_ctx._contributes[2];
	contrib[3] = m_ctx._contributes[3];
}

template<typename T, typename Tn>
const Vector3F & ClosestToPointEngine<T, Tn>::closestToPointPoint() const
{ return m_ctx._hitPoint; }

template<typename T, typename Tn>
const Vector3F & ClosestToPointEngine<T, Tn>::closestToPointNormal() const
{ return m_ctx._hitNormal; }

template<typename T, typename Tn>
Float2 ClosestToPointEngine<T, Tn>::closestToPointTexcoord() const
{
	const sdb::VectorArray<T> & src = SelectEngine<T, Tn>::source();
	const T * comp = src[m_ctx._isource];
	return comp->interpolateTexcoord(m_ctx._contributes);
}

template<typename T, typename Tn>
void ClosestToPointEngine<T, Tn>::getClosestToPointGeomcomp(int & geom, int & comp) const
{ 
	geom = m_ctx._igeometry;
	comp = m_ctx._icomponent;
}

template<typename T, typename Tn>
Plane ClosestToPointEngine<T, Tn>::closestPlane() const
{ return m_ctx.asPlane(); }

template<typename T, typename Tn>
Vector3F ClosestToPointEngine<T, Tn>::closestToPointColor() const
{
	const sdb::VectorArray<T> & src = SelectEngine<T, Tn>::source();
	const T * comp = src[m_ctx._isource];

	return comp->interpolateColor(m_ctx._contributes);
}

template<typename T, typename Tn>
int ClosestToPointEngine<T, Tn>::closestToPointColorCompressed() const
{ 
	const Vector3F c = closestToPointColor();
	int ic;
	col32::encodeC(ic, c.x, c.y, c.z, 1.f);
	return ic;
}

}
#endif
