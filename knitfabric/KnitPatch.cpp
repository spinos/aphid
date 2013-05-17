#include "KnitPatch.h"
#include <Vector3F.h>
#include <Vector2F.h>

static float yarmU[14] = {.1f, .2f, .4f, .25f, .15f, .11f, .33f, .66f, .89f, .85f, .75f, .6f, .8f, .9f };
static float yarmV[14] = {0.f, 0.f, .15f, .33f, .45f, .69f, .88f, .88f, .69f, .45f, .33f, .15f, 0.f, 0.f };
static float yarmW[14] = {-.3f, -.3f, .1f, .23f, .32f, .1f, -.5f, -.5f, .1f, .32f, .23f, .1f, -.3f, -.3f };
KnitPatch::KnitPatch() 
{
	m_indices = 0;
	m_yarnP = 0;
	m_thickness = 1.f;
}

KnitPatch::~KnitPatch() 
{
	delete[] m_indices;
	delete[] m_yarnP;
}

unsigned KnitPatch::numPointsPerGrid() const
{
    return 14;
}

unsigned KnitPatch::numPointsPerYarn() const
{
    return numPointsPerGrid() * m_numSeg;
}

void KnitPatch::setNumSeg(int num)
{
	if(m_indices) delete[] m_indices;
	if(m_yarnP) delete[] m_yarnP;
	m_numSeg = num;
	m_numYarn = num * 2;
	const unsigned nppy = numPointsPerYarn();
	m_yarnP = new Vector3F[m_numYarn * nppy];
	m_indices = new unsigned[nppy];
	for(unsigned i = 0; i < nppy; i++) m_indices[i] = i;
}

void KnitPatch::setThickness(float thickness)
{
	m_thickness = thickness;
}

Vector3F * KnitPatch::yarn()
{
    return m_yarnP;
}

Vector3F * KnitPatch::yarnAt(unsigned idx)
{
	return m_yarnP + idx * numPointsPerYarn();
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

void KnitPatch::createYarn(const Vector3F * tessellateP, const Vector3F * tessellateN)
{
	const unsigned nseg1 = m_numSeg + 1;
	float disp;
	Vector3F gN, dLat0, dLat1;
	unsigned gu0, gv0, gu1, gv1;
	unsigned i, j, k;
	unsigned yarnBegin = 0;
	unsigned firstBegin, secondBegin;
	if(m_uMajor) {
		for(vStart(j); vEnd(j); proceedV(j)) {
			gv0 = j;
			gv1 = j + m_vStep;
			firstBegin = yarnBegin;
			secondBegin = yarnBegin + numPointsPerYarn();
			for(uStart(i); uEnd(i); proceedU(i)) {
			
				gu0 = i;
				gu1 = i + m_uStep;
				
				gN = tessellateN[gv0 * nseg1 + gu0];
				
				setCorner(tessellateP[gv0 * nseg1 + gu0], 0);
				setCorner(tessellateP[gv0 * nseg1 + gu1], 1);
				setCorner(tessellateP[gv1 * nseg1 + gu1], 2);
				setCorner(tessellateP[gv1 * nseg1 + gu0], 3);
				
				dLat0 = tessellateP[gv1 * nseg1 + gu0] - tessellateP[gv0 * nseg1 + gu0];
				dLat1 = tessellateP[gv1 * nseg1 + gu1] - tessellateP[gv0 * nseg1 + gu1];
				dLat0 *= 0.5f;
				dLat1 *= 0.5f;
				
				for(k = 0; k < numPointsPerGrid(); k++) {
					m_yarnP[k + firstBegin] = interpolate(yarmU[k], yarmV[k]);
					disp = yarmW[k] * m_thickness;
					m_yarnP[k + firstBegin] += gN * disp;
				}
				
				/*
				for(k = 0; k < numPointsPerGrid(); k++) {
					m_yarnP[k + yarnBegin] = interpolate(yarmU[k], yarmV[k]);
					disp = yarmW[k] * m_thickness;
					m_yarnP[k + yarnBegin] += gN * disp;
				}
				yarnBegin += numPointsPerGrid();*/
				
				moveCorner(dLat0, 0);
				moveCorner(dLat1, 1);
				moveCorner(dLat1, 2);
				moveCorner(dLat0, 3);
				
				for(k = 0; k < numPointsPerGrid(); k++) {
					m_yarnP[k + secondBegin] = interpolate(yarmU[k], yarmV[k]);
					disp = yarmW[k] * m_thickness;
					m_yarnP[k + secondBegin] += gN * disp;
				}
				
				firstBegin += numPointsPerGrid();
				
				secondBegin += numPointsPerGrid();
			}
			yarnBegin += numPointsPerYarn() * 2;
		}
	}
	else {
		for(uStart(j); uEnd(j); proceedU(j)) {
			gu0 = j;
			gu1 = j + m_uStep;
			firstBegin = yarnBegin;
			secondBegin = yarnBegin + numPointsPerYarn();	
			for(vStart(i); vEnd(i); proceedV(i)) {
			
				gv0 = i;
				gv1 = i + m_vStep;
				
				gN = tessellateN[gv0 * nseg1 + gu0];
				
				setCorner(tessellateP[gv0 * nseg1 + gu0], 0);
				setCorner(tessellateP[gv1 * nseg1 + gu0], 1);
				setCorner(tessellateP[gv1 * nseg1 + gu1], 2);
				setCorner(tessellateP[gv0 * nseg1 + gu1], 3);
				
				dLat0 = tessellateP[gv0 * nseg1 + gu1] - tessellateP[gv0 * nseg1 + gu0];
				dLat1 = tessellateP[gv1 * nseg1 + gu1] - tessellateP[gv1 * nseg1 + gu0];
				dLat0 *= 0.5f;
				dLat1 *= 0.5f;
				
				for(k = 0; k < numPointsPerGrid(); k++) {
					m_yarnP[k + firstBegin] = interpolate(yarmU[k], yarmV[k]);
					disp = yarmW[k] * m_thickness;
					m_yarnP[k + firstBegin] += gN * disp;
				}
				
				/*
				for(k = 0; k < numPointsPerGrid(); k++) {
					m_yarnP[k + yarnBegin] = interpolate(yarmU[k], yarmV[k]);
					disp = yarmW[k] * m_thickness;
					m_yarnP[k + yarnBegin] += gN * disp;
				}
				yarnBegin += numPointsPerGrid();*/
				firstBegin += numPointsPerGrid();
				
				moveCorner(dLat0, 0);
				moveCorner(dLat1, 1);
				moveCorner(dLat1, 2);
				moveCorner(dLat0, 3);
				
				for(k = 0; k < numPointsPerGrid(); k++) {
					m_yarnP[k + secondBegin] = interpolate(yarmU[k], yarmV[k]);
					disp = yarmW[k] * m_thickness;
					m_yarnP[k + secondBegin] += gN * disp;
				}
				
				secondBegin += numPointsPerGrid();
			}
			yarnBegin += numPointsPerYarn() * 2;
		}
	}
}
