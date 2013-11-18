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
#include <MlCalamus.h>
#include <Patch.h>

class AccPatchMesh;
class MeshTopology;
class MlCalamusArray;
class BaseImage;

class MlSkin : public CollisionRegion 
{
public:
	MlSkin();
	virtual ~MlSkin();
	void cleanup();
	void clearFeather();
	
	void setNumFeathers(unsigned num);
	
	void setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo);
	void floodAround(MlCalamus c, unsigned idx, const Vector3F & pos, const Vector3F & nor, const float & maxD, const float & minD);
	void selectAround(unsigned idx, const Vector3F & pos, const Vector3F & nor, const float & maxD);
	void selectRegion(unsigned idx, const Vector2F & patchUV);
	void discardActive();
	
	void growFeather(const Vector3F & direction);
	void combFeather(const Vector3F & direction, const Vector3F & center, const float & radius);
	void scaleFeather(const Vector3F & direction, const Vector3F & center, const float & radius);
	void pitchFeather(const Vector3F & direction, const Vector3F & center, const float & radius);
	
	void computeFaceCalamusIndirection();
	void finishCreateFeather();
	void finishEraseFeather();
	unsigned numFeathers() const;
	MlCalamus * getCalamus(unsigned idx) const;
	
	AccPatchMesh * bodyMesh() const;
	unsigned numActive() const;
	MlCalamus * getActive(unsigned idx) const;
	
	void getPointOnBody(MlCalamus * c, Vector3F &p) const;
	void getNormalOnBody(MlCalamus * c, Vector3F &p) const;
	
	void tangentSpace(MlCalamus * c, Matrix33F & frm) const;
	void rotationFrame(MlCalamus * c, const Matrix33F & tang, Matrix33F & frm) const;
	
	virtual void resetCollisionRegion(unsigned idx);
	virtual void resetCollisionRegionAround(unsigned idx, const Vector3F & p, const float & d);
	virtual void closestPoint(const Vector3F & origin, IntersectionContext * ctx) const;
	virtual void pushPlane(Patch::PushPlaneContext * ctx) const;
	
	bool hasFeatherCreated() const;
	unsigned numCreated() const;
	MlCalamus * getCreated(unsigned idx) const;
	MlCalamusArray * getCalamusArray() const;
	
	void setFeatherDistributionMap(BaseImage * image);
	void verbose() const;
	
protected:
	
private:
	bool createFeather(MlCalamus & ori);
	bool isPointTooCloseToExisting(const Vector3F & pos, const unsigned faceIdx, float minDistance);
	bool isDartCloseToExisting(const Vector3F & pos, const std::vector<Vector3F> & existing, float minDistance) const;
	void resetFaceCalamusIndirection();
	unsigned lastInactive() const;
private:
	MlCalamusArray * m_calamus;
	std::vector<unsigned> m_activeIndices;
	unsigned m_numFeather, m_numCreatedFeather;
	AccPatchMesh * m_body;
	MeshTopology * m_topo;
	BaseImage * m_distribution;
	unsigned * m_faceCalamusStart;
};