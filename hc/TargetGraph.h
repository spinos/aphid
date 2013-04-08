/*
 *  TargetGraph.h
 *  hc
 *
 *  Created by jian zhang on 4/7/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <BaseMesh.h>
#include <BarycentricCoordinate.h>
#include <map>
class TargetGraph : public BaseMesh {
public:
	TargetGraph();
	virtual ~TargetGraph();
	void createVertexWeights(unsigned num);
	void createTargetIndices(unsigned num);
	void setTarget(unsigned idx, unsigned a);
	void setControlId(unsigned idx);
	void initCoords();
	void reset();
	void computeWeight(unsigned faceIdx, const Vector3F & pos);
	Vector3F getHandlePos() const;
	
	unsigned firstDirtyTarget();
	unsigned nextDirtyTarget();
	bool hasDirtyTarget();
	
	unsigned getControlId() const;
	float targetWeight(unsigned idx) const;

private:
	void addDirtyTargets(unsigned faceIdx);
	std::map<unsigned, unsigned> m_dirtyTargets;
	std::map<unsigned, unsigned>::iterator m_dirtyTargetIt;
	unsigned m_controlId;
	float * m_vertexWeights;
	unsigned * m_targetIndices;
	BarycentricCoordinate * m_baryc;
	Vector3F m_handlePos;
	unsigned m_previousFace;
};