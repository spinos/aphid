#include "KnitPatch.h"
#include <Vector3F.h>
#include <Vector2F.h>

static float yarmU[14] = {0.f, .2f, .4f, .4f, .1f, .1f, .33f, .66f, .9f, .9f, .6f, .6f, .8f, 1.f };
static float yarmV[14] = {0.f, 0.f, .15f, .33f, .45f, .67f, .8f, .8f, .67f, .45f, .33f, .15f, 0.f, 0.f };
KnitPatch::KnitPatch() {}
KnitPatch::~KnitPatch() 
{
	delete[] m_indices;
	delete[] m_yarnP;
}

unsigned KnitPatch::numYarnPointsPerGrid() const
{
    return 14;
}

unsigned KnitPatch::numYarnPoints() const
{
    return numYarnPointsPerGrid() * m_numSeg;
}

void KnitPatch::setNumSeg(int num)
{
	m_numSeg = num;
	m_numYarn = num;
	const unsigned np = numYarnPointsPerGrid();
	m_yarnP = new Vector3F[num * num * np];
	m_indices = new unsigned[num * np];
	for(unsigned i = 0; i < num * np; i++) m_indices[i] = i;
}

Vector3F * KnitPatch::yarn()
{
    return m_yarnP;
}

unsigned * KnitPatch::yarnIndices()
{
	return m_indices;
}
#include <iostream>
void KnitPatch::directionByBiggestDu(Vector2F *uv)
{
    unsigned r = 0;
    float maxdu = -10e8;
    for(unsigned i = 0; i < 4; i++) {
        unsigned i1 = i + 1;
        if(i1 > 3) i1 = 0;
        float du = uv[i1].x - uv[i].x;
		//printf("%i du %f ", i, du);
        if(du > maxdu) {
            maxdu = du;
            r = i;
        }
    }
	//if(r != 0) printf("%i %f ", r, maxdu);

	if(r == 0) {
		m_uMajor = 1;
		m_uGridMin = 0;
		m_uGridMax = m_numSeg;
		m_vGridMin = 0;
		m_vGridMax = m_numSeg;
		m_uStep = 1;
		m_vStep = 1;
	}
	else if(r == 1) {
		m_uMajor = 0;
		m_uGridMin = m_numSeg;
		m_uGridMax = 0;
		m_vGridMin = 0;
		m_vGridMax = m_numSeg;
		m_uStep = -1;
		m_vStep = 1;
	}
	else if(r == 2) {
		m_uMajor = 1;
		m_uGridMin = m_numSeg;
		m_uGridMax = 0;
		m_vGridMin = m_numSeg;
		m_vGridMax = 0;
		m_uStep = -1;
		m_vStep = -1;
	}
	else {
		m_uMajor = 0;
		m_uGridMin = 0;
		m_uGridMax = m_numSeg;
		m_vGridMin = m_numSeg;
		m_vGridMax = 0;
		m_uStep = 1;
		m_vStep = -1;
	}
}

unsigned KnitPatch::getNumYarn() const
{
	return m_numYarn;
}

void KnitPatch::uStart(unsigned &val) const
{
	val = m_uGridMin;
}

void KnitPatch::vStart(unsigned &val) const
{
	val = m_vGridMin;
}

bool KnitPatch::uEnd(unsigned val) const
{
	if(m_uStep > 0) return val < m_uGridMax;
	return val > m_uGridMax;
}

bool KnitPatch::vEnd(unsigned val) const
{
	if(m_vStep > 0) return val < m_vGridMax;
	return val > m_vGridMax;
}

void KnitPatch::proceedU(unsigned &val) const
{
	val += m_uStep;
}

void KnitPatch::proceedV(unsigned &val) const
{
	val += m_vStep;
}

void KnitPatch::createYarn(const Vector3F * tessellateP)
{
	unsigned gu0, gv0, gu1, gv1;
	unsigned i, j, k;
	unsigned yarnBegin = 0;
	if(m_uMajor) {
		for(vStart(j); vEnd(j); proceedV(j)) {
			gv0 = j;
			gv1 = j + m_vStep;
			for(uStart(i); uEnd(i); proceedU(i)) {
			
				gu0 = i;
				gu1 = i + m_uStep;
				
				setCorner(tessellateP[gv0 * (m_numSeg + 1) + gu0], 0);
				setCorner(tessellateP[gv0 * (m_numSeg + 1) + gu1], 1);
				setCorner(tessellateP[gv1 * (m_numSeg + 1) + gu1], 2);
				setCorner(tessellateP[gv1 * (m_numSeg + 1) + gu0], 3);
				
				for(k = 0; k < numYarnPointsPerGrid(); k++) {
					m_yarnP[k + yarnBegin] = interpolate(yarmU[k], yarmV[k]);
				}
				yarnBegin += numYarnPointsPerGrid();
			}
		}
	}
	else {
		for(uStart(j); uEnd(j); proceedU(j)) {
			gu0 = j;
			gu1 = j + m_uStep;
			for(vStart(i); vEnd(i); proceedV(i)) {
			
				gv0 = i;
				gv1 = i + m_vStep;
				
				setCorner(tessellateP[gv0 * (m_numSeg + 1) + gu0], 0);
				setCorner(tessellateP[gv1 * (m_numSeg + 1) + gu0], 1);
				setCorner(tessellateP[gv1 * (m_numSeg + 1) + gu1], 2);
				setCorner(tessellateP[gv0 * (m_numSeg + 1) + gu1], 3);
				
				for(k = 0; k < numYarnPointsPerGrid(); k++) {
					m_yarnP[k + yarnBegin] = interpolate(yarmU[k], yarmV[k]);
				}
				yarnBegin += numYarnPointsPerGrid();
			}
		}
	}
}
