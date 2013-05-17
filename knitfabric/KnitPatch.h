#pragma once

#include <LODQuad.h>
class Vector2F;

class KnitPatch : public LODQuad {
public:
    KnitPatch();
    virtual ~KnitPatch();
    unsigned numPointsPerYarn() const;
	unsigned numPointsPerGrid() const;
    void setNumSeg(int num);
	void setThickness(float thickness);
	
	void createYarn(const Vector3F * tessellateP, const Vector3F * tessellateN);
    
    Vector3F * yarn();
	Vector3F * yarnAt(unsigned idx);
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
	
	unsigned * m_indices;
    Vector3F * m_yarnP;
	unsigned m_numSeg;
	unsigned m_numYarn;
	unsigned m_uGridMin, m_uGridMax;
	unsigned m_vGridMin, m_vGridMax;
	int m_uStep, m_vStep;
	float m_thickness;
	char m_uMajor;
};

