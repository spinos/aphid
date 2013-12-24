#pragma once
#include "BaseVane.h"
class MlVane : public BaseVane {
public:
    MlVane();
    virtual ~MlVane();
	
	virtual void setU(float u);
	
	void setSeed(unsigned s);
	void separate(unsigned nsep);
	void setSeparateStrength(float k);
	void setFuzzy(float f);
	void modifyLength(float u, unsigned gridV, Vector3F * dst);
    
private:
	void clear();
	void computeSeparation();
	void computeLengthChange();
	float getSeparateU(float u, float * param) const;
	void setU(float u0, float u1);
private:
	unsigned m_numSeparate;
	float * m_barbBegin;
	float * m_separateEnd;
	float * m_lengthChange;
	unsigned m_seed;
	float m_separateStrength, m_fuzzy;
};
