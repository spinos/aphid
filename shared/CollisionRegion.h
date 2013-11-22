/*
 *  CollisionRegion.h
 *  mallard
 *
 *  Created by jian zhang on 9/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <Patch.h>
#include <LineBuffer.h>
class IntersectionContext;
class MeshTopology;
class AccPatchMesh;
class BaseImage;
class CollisionRegion : public LineBuffer {
public:
	CollisionRegion();
	virtual ~CollisionRegion();
	void clearCollisionRegion();
	
	AccPatchMesh * bodyMesh() const;
	
	Vector3F getClosestPoint(const Vector3F & origin);
	
	virtual void setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo);
	virtual void resetCollisionRegion(unsigned idx);
	virtual void resetCollisionRegionAround(unsigned idx, const Vector3F & p, const float & d);
	virtual void closestPoint(const Vector3F & origin, IntersectionContext * ctx) const;
	virtual void pushPlane(Patch::PushPlaneContext * ctx) const;
	
	unsigned numRegionElements() const;
	unsigned regionElementIndex(unsigned idx) const;
	
	unsigned regionElementStart() const;
	void setRegionElementStart(unsigned x);
	
	void setDistributionMap(BaseImage * image);
	void selectRegion(unsigned idx, const Vector2F & patchUV);
	
	std::vector<unsigned> * regionElementIndices();
	
	char faceColorMatches(unsigned idx) const;
	
	virtual void rebuildBuffer();
	
	void neighborFaces(unsigned idx, std::vector<unsigned> & dst);
	char sampleColorMatches(unsigned idx, float u, float v) const;
	Vector3F sampleColor() const;
	void colorAt(unsigned idx, float u, float v, Vector3F * dst) const;
private:
    void fillPatchEdge(unsigned iface, unsigned iedge, unsigned vstart);
	MeshTopology * m_topo;
	AccPatchMesh * m_body;
	std::vector<unsigned> m_regionElementIndices;
	unsigned m_regionElementStart;
	BaseImage * m_distribution;
	IntersectionContext * m_ctx;
	Vector3F m_sampleColor;
};