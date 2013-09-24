/*
 *  MlSkin.h
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <CollisionRegion.h>

class AccPatchMesh;
class MeshTopology;
class MlCalamus;
class MlCalamusArray;

class MlSkin : public CollisionRegion {
public:
	MlSkin();
	virtual ~MlSkin();
	void setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo);
	void floodAround(MlCalamus c, unsigned idx, const Vector3F & pos, const float & maxD, const float & minD);
	void selectAround(unsigned idx, const Vector3F & pos, const float & maxD);
	void discardActive();
	
	void growFeather(const Vector3F & direction);
	void finishCreateFeather();
	void finishEraseFeather();
	unsigned numFeathers() const;
	MlCalamus * getCalamus(unsigned idx) const;
	
	AccPatchMesh * bodyMesh() const;
	unsigned numActive() const;
	MlCalamus * getActive(unsigned idx) const;
	
	void getPointOnBody(MlCalamus * c, Vector3F &p) const;
	Matrix33F tangentFrame(MlCalamus * c) const;
	
	virtual void resetCollisionRegion(unsigned idx);
	virtual void resetCollisionRegionAround(unsigned idx, const Vector3F & p, const float & d);
	virtual void closestPoint(const Vector3F & origin, IntersectionContext * ctx) const;
	
	bool hasFeatherCreated() const;
	unsigned numCreated() const;
	MlCalamus * getCreated(unsigned idx) const;
	void verbose() const;
private:
	bool createFeather(MlCalamus & ori);
	bool isPointTooCloseToExisting(const Vector3F & pos, const unsigned faceIdx, float minDistance);
	bool isDartCloseToExisting(const Vector3F & pos, const std::vector<Vector3F> & existing, float minDistance) const;
	void computeFaceCalamusIndirection();
	void resetFaceCalamusIndirection();
	unsigned lastInactive() const;
private:
	MlCalamusArray * m_calamus;
	std::vector<unsigned> m_activeIndices;
	unsigned m_numFeather, m_numCreatedFeather;
	AccPatchMesh * m_body;
	MeshTopology * m_topo;
	unsigned * m_faceCalamusStart;
};