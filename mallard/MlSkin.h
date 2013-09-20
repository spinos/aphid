/*
 *  MlSkin.h
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <CollisionRegion.h>
#include "MlCalamusArray.h"

class AccPatchMesh;
class MeshTopology;
class MlSkin : public CollisionRegion {
public:
	MlSkin();
	virtual ~MlSkin();
	void setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo);
	bool createFeather(MlCalamus & ori, const Vector3F & pos, float minDistance);
	void growFeather(const Vector3F & direction);
	void finishCreateFeather();
	unsigned numFeathers() const;
	MlCalamus * getCalamus(unsigned idx) const;
	
	AccPatchMesh * bodyMesh() const;
	unsigned numActiveFeather() const;
	MlCalamus * getActive(unsigned idx) const;
	
	void getPointOnBody(MlCalamus * c, Vector3F &p) const;
	Matrix33F tangentFrame(MlCalamus * c) const;
	
	virtual void resetCollisionRegion(unsigned idx);
	
	void verbose() const;
private:
	bool isPointTooCloseToExisting(const Vector3F & pos, const unsigned faceIdx, float minDistance);
	
private:
	MlCalamusArray m_calamus;
	std::vector<unsigned> m_activeIndices;
	unsigned m_numFeather;
	AccPatchMesh * m_body;
	MeshTopology * m_topo;
	unsigned * m_faceCalamusStart;
};