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
#include "CalamusSkin.h"

class AccPatchMesh;
class MeshTopology;
class MlCalamusArray;
class BaseImage;
class SelectCondition;
class FloodCondition;
class MlCluster;

class MlSkin : public CalamusSkin 
{
public:
	
	MlSkin();
	virtual ~MlSkin();

	virtual void setBodyMesh(AccPatchMesh * mesh, MeshTopology * topo);
	
	void floodAround(MlCalamus c, FloodCondition * condition);
	void selectAround(unsigned idx, SelectCondition * selcon);
	void discardActive();
	
	void growFeather(const Vector3F & direction);
	void combFeather(const Vector3F & direction);
	void scaleFeather(const Vector3F & direction);
	void pitchFeather(const Vector3F & direction);
	void smoothShell(const Vector3F & center, const float & radius, const float & weight);
	void computeVertexDisplacement();
	void finishCreateFeather();
	void finishEraseFeather();

	unsigned numActive() const;
	MlCalamus * getActive(unsigned idx) const;
	
	bool hasFeatherCreated() const;
	unsigned numCreated() const;
	MlCalamus * getCreated(unsigned idx) const;
	
	void resetFloodFaces();
	void restFloodFacesAsActive();
	
	void shellUp(std::vector<Vector3F> & dst);
	
	void verbose() const;
protected:
	
private:
    bool createFeather(MlCalamus & ori);
	bool isDartCloseToExisting(const Vector3F & pos, const std::vector<Vector3F> & existing, float minDistance) const;
	bool isFloodFace(unsigned idx, unsigned & dartBegin, unsigned & dartEnd) const;
	bool isActiveFace(unsigned idx, std::vector<unsigned> & dartIndices) const;
	bool isActiveFeather(unsigned idx) const;
	unsigned lastInactive(unsigned last) const;
	unsigned selectFeatherByFace(unsigned faceIdx, SelectCondition * selcon);
	void computeAffectWeight(const Vector3F & center, const float & radius);
private:	
	std::vector<unsigned> m_activeIndices;
	unsigned m_numCreatedFeather;
	std::vector<FloodTable> m_activeFaces;
	std::vector<FloodTable> m_floodFaces;
	float * m_affectWeights;
};