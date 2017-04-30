#include "CurveReduction.h"
#include <GeometryArray.h>
#include <BezierCurve.h>
#include <CurveBuilder.h>

CurveReduction::CurveReduction() 
{
	m_gapHash = 0;
}

CurveReduction::~CurveReduction() 
{
	if(m_gapHash) delete[] m_gapHash;
} 

GeometryArray *  CurveReduction::compute(GeometryArray * curves, float distanceThreshold)
{
    const unsigned n = curves->numGeometries();
    if(n < 2) {
        std::cout<<" curve reduction insufficient n curves "<<n<<".\n";
        return 0;
    }
    
	Vector3F * samples = new Vector3F[n];
// sample curve starts
    unsigned i = 0;
    for(;i<n;i++) {
        BaseCurve * c = static_cast<BaseCurve *>(curves->geometry(i));
        samples[i] = c->m_cvs[0];
    }
	
	computeSampleGaps(samples, n, curves);
	
	int m = computeM(n, distanceThreshold);
	if(m<1) {
		// std::cout<<" curve reduction finds no curves close enough, skipped.\n";
		delete[] samples;
		return 0;
	}
	
	std::cout<<" reduce curves from "<<n<<" to "<<n-m<<".\n";
	
    KMeanSampleGroup::compute(samples, n, n - m);
    delete[] samples;
    
    GeometryArray * merged = mergeCurves(curves);
    return merged;
}

void CurveReduction::computeSampleGaps(Vector3F * samples, unsigned n, GeometryArray * curves)
{
	if(m_gapHash) delete[] m_gapHash;
	m_gapHash = new GapInd[n];
	
	Vector3F p;
	float d;
	unsigned i, j;
	for(i=0; i<n; i++) {
		m_gapHash[i].value = i;
		m_gapHash[i].key = 1e8f;
		p = samples[i];
		for(j=0; j<n; j++) {
			if(j==i) continue;
			d = p.distanceTo(samples[j]);
			if(m_gapHash[i].key > d) m_gapHash[i].key = d;
		}
	}
	
	QuickSort1::Sort<float, unsigned >(m_gapHash, 0, n-1);
    
}

int CurveReduction::computeM(unsigned n, float distanceThreshold)
{
	int m = 0;
	unsigned i;
	for(i=0; i<n; i++) { 
		if(m_gapHash[i].key <= distanceThreshold) m++;
		// std::cout<<"\n gap "<<m_gapHash[i].key<<" s"<<m_gapHash[i].value; 
	}

	return m/2;
}

void CurveReduction::initialGuess(const Vector3F * pos)
{
// sample group center start from the one with largest gap
	unsigned i, s;
    for(i=0; i< K(); i++) {
		s = m_gapHash[N() - 1 - i].value;
		setCentroid(i, pos[s]);
	}
}

GeometryArray * CurveReduction::mergeCurves(GeometryArray * curves)
{
    GeometryArray * outGeo = new GeometryArray;
    outGeo->create(K());
    unsigned i;
    for(i=0; i< K(); i++) {
        if(countPerGroup(i)>1) {
            outGeo->setGeometry(mergeCurvesInGroup(i, curves), i);
        }
        else {
            outGeo->setGeometry(duplicateCurve(i, curves), i);
        }
    }
    return outGeo;
}

unsigned CurveReduction::minimumNumCvsInGroup(unsigned igroup, GeometryArray * curves) const
{
    unsigned res = 9999;
    unsigned i;
    for(i=0; i< N(); i++) {
        if(group(i) == igroup) {
            BaseCurve * c = static_cast<BaseCurve *>(curves->geometry(i));
            if(res > c->numVertices()) res = c->numVertices();
        }
    }
    return res;
}

unsigned CurveReduction::maximumNumCvsInGroup(unsigned igroup, GeometryArray * curves) const
{
    unsigned res = 1;
    unsigned i;
    for(i=0; i< N(); i++) {
        if(group(i) == igroup) {
            BaseCurve * c = static_cast<BaseCurve *>(curves->geometry(i));
            if(res < c->numVertices()) res = c->numVertices();
        }
    }
    return res;
}

void CurveReduction::weightByCurveLengths(float * weights, unsigned igroup, GeometryArray * curves) const
{
	float sum = 0.f;
	unsigned i;
    for(i=0; i< N(); i++) {
        if(group(i) == igroup) {
            BaseCurve * c = static_cast<BaseCurve *>(curves->geometry(i));
            weights[i] = c->length();
			sum += weights[i];
        }
    }
	
	for(i=0; i< N(); i++) weights[i] /= sum;
}

BezierCurve * CurveReduction::duplicateCurve(unsigned igroup, GeometryArray * curves)
{
    CurveBuilder cb;
    unsigned i=0;
    
    BaseCurve * src;
    for(i=0; i< N(); i++) {
        if(group(i) == igroup) {
            src = static_cast<BaseCurve *>(curves->geometry(i));
            break;
        }
    }
    
    for(i=0;i<src->numVertices();i++)
        cb.addVertex(src->m_cvs[i]);
    
    BezierCurve * b = new BezierCurve;
    cb.finishBuild(b);
    return b;
}

BezierCurve * CurveReduction::mergeCurvesInGroup(unsigned igroup, GeometryArray * geos)
{
	float * weights = new float[N()];
	weightByCurveLengths(weights, igroup, geos);
    unsigned nv = maximumNumCvsInGroup(igroup, geos);
    const float delta = 1.f / (float)(nv-1);
    CurveBuilder cb;
    unsigned i=0;
    for(;i< nv;i++)
        cb.addVertex(samplePointsInGroup(delta * i, igroup, geos, weights));
    
	delete[] weights;
    BezierCurve * b = new BezierCurve;
    cb.finishBuild(b);
    return b;
}

Vector3F CurveReduction::samplePointsInGroup(float param, unsigned igroup, GeometryArray * geos,
											float * weights)
{
	const float meanWeight = 1.f / (float)countPerGroup(igroup);
	
    Vector3F sum = Vector3F::Zero;
    unsigned i = 0;
    for(; i< N(); i++) {
        if(group(i) == igroup) {
            BaseCurve * c = static_cast<BaseCurve *>(geos->geometry(i));
            sum += c->interpolate(param) * ( meanWeight * (1.f - param) + weights[i] * param );
        }
    }
    
    return sum;
}
//:~