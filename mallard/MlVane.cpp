/*
 *    V      L
 *    |      |
 *    c -----|-----
 *    |      |
 *    c -----|----- U
 *
 */

#include "MlVane.h"
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

void MlVane::create(unsigned gU, unsigned gV)
{
	BaseVane::create(gU, gV);
	createPlot(2048);
	computeNoise();
}

void MlVane::setSeed(unsigned s)
{
	m_seed = s;
}

void MlVane::setNumSparate(unsigned nsep)
{
	clear();
	if(nsep < 1) return;
	m_numSeparate = nsep * gridU();
	m_barbBegin = new float[m_numSeparate];
	m_separateEnd = new float[m_numSeparate * 2];
	m_lengthChange = new float[m_numSeparate * 2];
}

void MlVane::separate()
{
	computeSeparation();
	computeLengthChange();
}

void MlVane::computeSeparation()
{
	const float ds = 1.f / m_numSeparate;
	float r;
	for(unsigned i = 0; i < m_numSeparate; i++) {
		m_barbBegin[i] = ds * i;
	}
	
	for(unsigned i = 0; i < m_numSeparate; i++) {
		r = (sample(m_seed + i * 13) + .5f)* 2.f - 1.f;
		m_separateEnd[i*2] = m_barbBegin[i] + ds * r * 2.f * m_separateStrength;
		
		if(m_separateEnd[i*2]< 0.f) m_separateEnd[i*2] = 0.f;
		else if(m_separateEnd[i*2] > 1.f) m_separateEnd[i*2] = 1.f;
		
		r = sample(m_seed + i * 15);
		m_separateEnd[i*2 + 1] = m_separateEnd[i*2] + ds * (1.f + r * m_separateStrength);
		
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
	const float ds = 1.f /(float)m_numSeparate;
	unsigned i = u / ds;
	float portion = (u - i * ds)/ds;
	*param = i + portion;
	return m_separateEnd[i * 2] + (m_separateEnd[i * 2 + 1] - m_separateEnd[i * 2]) * portion;
}

void MlVane::modifyLength(float u, unsigned gridV, Vector3F * dst, float lod)
{
	if(u == 1.f) return;
	float param;
	getSeparateU(u, &param);
	const int barb = (int)param;
	const float port = param - barb;
	const float dl = m_lengthChange[barb * 2] * (1.f - port) + m_lengthChange[barb * 2 + 1] * port;
	Vector3F dp;
	float wei;
	const unsigned freq = 16 * gridU();
	for(unsigned i = 1; i < gridV; i++) {
		dp = dst[i] - dst[i - 1];
		wei = dl;
		if(m_fuzzy > 0.f) wei += getNoise(u, freq, lod, m_seed) * m_fuzzy * .5f;

		dp *= wei;
		
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

void MlVane::computeNoise()
{
	computePlot(m_seed);
}

unsigned MlVane::seed() const
{
	return m_seed;
}