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
	m_barbBegin = 0;
	m_separateEnd = 0;
	m_numSeparate = 0;
	m_lengthChange = 0;
	m_separateStrength = 0.f;
	m_fuzzy = 0.f;
}

MlVane::~MlVane() 
{
	clear();
}

void MlVane::clear()
{
	if(m_barbBegin) delete[] m_barbBegin;
	if(m_separateEnd) delete[] m_separateEnd;
	if(m_lengthChange) delete[] m_lengthChange;
	m_barbBegin = 0;
	m_separateEnd = 0;
	m_lengthChange = 0;
}

void MlVane::setSeed(unsigned s)
{
	m_seed = s;
}

void MlVane::setNumSparate(unsigned nsep)
{
	clear();
	m_numSeparate = nsep;
	if(nsep < 2) return;
	m_barbBegin = new float[nsep];
	m_separateEnd = new float[nsep * 2];
	m_lengthChange = new float[nsep * 2];
}

void MlVane::separate()
{
	computeSeparation();
	computeLengthChange();
}

void MlVane::computeSeparation()
{
	m_barbBegin[0] = 0.f;
	PseudoNoise noi;
	
	const float ds = 1.f / m_numSeparate;
	float r;
	for(unsigned i = 1; i < m_numSeparate; i++) {
		r = noi.rfloat(m_seed + i * 19) * 0.7f;
		m_barbBegin[i] = ds * i + ds * r;
	}
	
	float barbW;
	for(unsigned i = 0; i < m_numSeparate; i++) {
		if(i < m_numSeparate - 1) barbW = m_barbBegin[i+1] - m_barbBegin[i];
		else barbW = 1.f - m_barbBegin[i];

		r = noi.rfloat(m_seed + i * 13) * 2.f - 1.f;
		m_separateEnd[i*2] = m_barbBegin[i] + barbW * r * 2.f * m_separateStrength;
		
		if(m_separateEnd[i*2]< 0.f) m_separateEnd[i*2] = 0.f;
		else if(m_separateEnd[i*2] > 1.f) m_separateEnd[i*2] = 1.f;
		
		r = noi.rfloat(m_seed + i * 15) - 0.5f;
		m_separateEnd[i*2 + 1] = m_separateEnd[i*2] + barbW * (1.f + r * m_separateStrength);
		
		if(m_separateEnd[i*2 + 1]< 0.f) m_separateEnd[i*2 + 1] = 0.f;
		else if(m_separateEnd[i*2 + 1] > 1.f) m_separateEnd[i*2 + 1] = 1.f;
	}
}

void MlVane::computeLengthChange()
{
	float l0, l1, barbEnd;
	unsigned i;
	for(i = 0; i < m_numSeparate; i++) {
		BaseVane::setU(m_barbBegin[i]);
		l0 = profile()->length();
		setU(m_barbBegin[i], m_separateEnd[i*2]);
		l1 = profile()->length();
		m_lengthChange[i*2] = l0 / l1 - 1.f; //std::cout<<"   "<<l0 / l1 - 1.f;
		
		if(i < m_numSeparate - 1) barbEnd = m_barbBegin[i + 1];
		else barbEnd = 1.f;

		BaseVane::setU(barbEnd);
		l0 = profile()->length();
		setU(barbEnd, m_separateEnd[i*2+1]);
		l1 = profile()->length();
		m_lengthChange[i*2+1] = l0 / l1 - 1.f; //std::cout<<" "<<l0 / l1 - 1.f;
	}
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
		if(u >= m_barbBegin[i] && u < m_barbBegin[i+1]) {
			portion = (u - m_barbBegin[i]) / (m_barbBegin[i+1] - m_barbBegin[i]);
			*param = i + portion;
			return m_separateEnd[i * 2] + (m_separateEnd[i * 2 + 1] - m_separateEnd[i * 2]) * portion;
		}
	}
	portion = (u - m_barbBegin[i]) / (1.f - m_barbBegin[i]);
	*param = i + portion;
	return m_separateEnd[i * 2] + (m_separateEnd[i * 2 + 1] - m_separateEnd[i * 2]) * portion;
}

void MlVane::modifyLength(float u, unsigned gridV, Vector3F * dst)
{
	if(u == 1.f) return;
	float param;
	getSeparateU(u, &param);
	const int barb = (int)param;
	const float port = param - barb;
	const float dl = m_lengthChange[barb * 2] * (1.f - port) + m_lengthChange[barb * 2 + 1] * port;
	PseudoNoise noi;
	Vector3F dp;
	float wei;
	for(unsigned i = 1; i < gridV; i++) {
		dp = dst[i] - dst[i - 1];
		wei = dl;
		if(m_fuzzy > 0.f)
			wei += (noi.rfloat(m_seed + u * 109493) - 0.5f) * m_fuzzy * .5f;

		dp *= wei; // if(u>0.98f)std::cout<<" "<<u<<" "<<dl;
		
		for(unsigned j = i; j <= gridV; j++) {
			dst[j] += dp;
		}
	}
}

void MlVane::setSeparateStrength(float k)
{
	m_separateStrength = k;
}

void MlVane::setFuzzy(float f)
{
	m_fuzzy = f;
}
