/*
 *    V      L
 *    |      |
 *    c -----|-----
 *    |      |
 *    c -----|----- U
 *
 */

#include "MlVane.h"
#include <PseudoNoise.h>
MlVane::MlVane() 
{
	m_separateBegin = 0;
	m_separateEnd = 0;
	m_numSeparate = 0;
	m_lengthChange = 0;
}

MlVane::~MlVane() 
{
	clear();
}

void MlVane::clear()
{
	if(m_separateBegin) delete[] m_separateBegin;
	if(m_separateEnd) delete[] m_separateEnd;
	if(m_lengthChange) delete[] m_lengthChange;
	m_separateBegin = 0;
	m_separateEnd = 0;
	m_lengthChange = 0;
}

void MlVane::setSeed(unsigned s)
{
	m_seed = s;
}

void MlVane::separate(unsigned nsep)
{
	clear();
	m_numSeparate = nsep;
	if(nsep < 2) return;
	m_separateBegin = new float[nsep];
	m_separateEnd = new float[nsep];
	m_lengthChange = new float[nsep];
	computeSeparation();
	computeLengthChange();
}

void MlVane::computeSeparation()
{
	m_separateBegin[0] = 0.f;
	PseudoNoise noi;
	
	float r;
	for(unsigned i = 1; i < m_numSeparate; i++) {
		r = noi.rfloat(m_seed + i * 19) * 0.5f + 0.5f;
		m_separateBegin[i] = m_separateBegin[i-1] + (1.f - m_separateBegin[i-1]) / ((m_numSeparate - i)/2 + 1)  * r;
	}
	
	for(unsigned i = 0; i < m_numSeparate - 1; i++) {
		r = (noi.rfloat(m_seed + i * 13) - 0.5f) * 1.8f + .9f;
		m_separateEnd[i] = (m_separateBegin[i + 1] - m_separateBegin[i]) * r;
		if(m_separateBegin[i] + m_separateEnd[i] > 1.f) m_separateEnd[i] = 1.f - m_separateBegin[i];
	}
	m_separateEnd[m_numSeparate - 1] = 1.f - m_separateBegin[m_numSeparate - 1];
}

void MlVane::computeLengthChange()
{
	float l0, l1;
	unsigned i;
	for(i = 0; i < m_numSeparate - 1; i++) {
		BaseVane::setU(m_separateBegin[i + 1]);
		l0 = profile()->length();
		setU(m_separateBegin[i + 1], m_separateBegin[i] + m_separateEnd[i]);
		l1 = profile()->length();
		m_lengthChange[i] = l1 / l0 - 1.f;
	}
	m_lengthChange[i] = 0.f;
}

void MlVane::setU(float u)
{
	if(m_numSeparate < 2) {
		BaseVane::setU(u);
		return;
	}
	
	float p;
	const float tu = getSeparateU(u, &p);
	setU(u, tu);
}

void MlVane::setU(float u0, float u1)
{
	profile()->m_cvs[0] = rails()[0].interpolate(u0);
	profile()->m_cvs[1] = rails()[1].interpolate(u0);
	float wei;
	for(unsigned i=2; i <= gridV(); i++) {
		wei = (float)(i - 1)/(float)(gridV() - 1);
        profile()->m_cvs[i] = rails()[i].interpolate(u0 + (u1 - u0) * wei);
    }
    profile()->computeKnots();
}

float MlVane::getSeparateU(float u, float * param) const
{
	unsigned i;
	float portion;
	for(i= 0; i < m_numSeparate - 1; i++) {
		if(u >= m_separateBegin[i] && u < m_separateBegin[i+1]) {
			portion = (u - m_separateBegin[i]) / (m_separateBegin[i+1] - m_separateBegin[i]);
			*param = i + portion;
			return m_separateBegin[i] + m_separateEnd[i] * portion;
		}
	}
	portion = (u - m_separateBegin[i]) / (1.f - m_separateBegin[i]);
	*param = i + portion;
	return m_separateBegin[i] + m_separateEnd[i] * portion;
}

void MlVane::modifyLength(float u, unsigned gridV, Vector3F * dst)
{
	float param;
	getSeparateU(u, &param);
	const float dl = m_lengthChange[(int)param] * (param - (int)param);
	if(dl < 10e-6) return;
	
	Vector3F dp;
	for(unsigned i = gridV / 2; i < gridV; i++) {
		dp = dst[i] - dst[i - 1];
		dp *= dl;
		for(unsigned j = i; j <= gridV; j++) {
			dst[j] += dp;
		}
	}
}

