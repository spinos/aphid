/*
 *  KnitPatch.h
 *  knitfabric
 *
 *  Created by jian zhang on 11/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <BaseQuad.h>

class KnitPatch : public BaseQuad {
public:
    KnitPatch();
    virtual ~KnitPatch();
    void cleanup();
    unsigned numPointsPerYarn() const;
	unsigned numPointsPerGrid() const;
    void setNumSeg(int num);
	void setThickness(float thickness);
	
	void createYarn(const Vector3F * tessellateP, const Vector3F * tessellateN);
    
    Vector3F * yarn();
	Vector3F * yarnAt(unsigned idx);
	Vector3F * normalAt(unsigned idx);
	Vector3F * tangentAt(unsigned idx);
	unsigned * yarnIndices();
	unsigned getNumYarn() const;
	
	void directionByBiggestDu(Vector2F *uv);
private:
	void uStart(unsigned &val) const;
	void vStart(unsigned &val) const;
	bool uEnd(unsigned val) const;
	bool vEnd(unsigned val) const;
	void proceedU(unsigned &val) const;
	void proceedV(unsigned &val) const;
	void calculateTangent();
	
	unsigned * m_indices;
    Vector3F * m_yarnP;
    Vector3F * m_yarnN;
    Vector3F * m_yarnT;
	unsigned m_numSeg;
	unsigned m_numYarn;
	unsigned m_uGridMin, m_uGridMax;
	unsigned m_vGridMin, m_vGridMax;
	int m_uStep, m_vStep;
	float m_thickness;
	char m_uMajor;
};

