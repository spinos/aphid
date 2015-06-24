#include "CurveReduction.h"
#include <GeometryArray.h>
#include <BezierCurve.h>
#include <CurveBuilder.h>

CurveReduction::CurveReduction() 
{
}

CurveReduction::~CurveReduction() 
{
} 

GeometryArray *  CurveReduction::compute(GeometryArray * curves, float alpha)
{
    const unsigned n = curves->numGeometries();
    int m = n * alpha;
    m = n - n/2;
    if(m<1) m = 1;
    if((int)n - m < 1) {
        std::cout<<" curve reduction insufficient n curves "<<n<<".\n";
        return 0;
    }
    std::cout<<" curve reduce curves from "<<n<<" to "<<n-m<<".\n";
    Vector3F * samples = new Vector3F[n];
    
    unsigned i = 0;
    for(;i<n;i++) {
        BaseCurve * c = static_cast<BaseCurve *>(curves->geometry(i));
        samples[i] = c->m_cvs[0];
    }
    
    KMeanSampleGroup::compute(samples, n, n - m);
    delete[] samples;
    
    GeometryArray * merged = mergeCurves(curves);
    return merged;
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
    unsigned nv = minimumNumCvsInGroup(igroup, geos);
    const float delta = 1.f / (float)(nv-1);
    CurveBuilder cb;
    unsigned i=0;
    for(;i< nv;i++)
        cb.addVertex(samplePointsInGroup(delta * i, igroup, geos));
    
    BezierCurve * b = new BezierCurve;
    cb.finishBuild(b);
    return b;
}

Vector3F CurveReduction::samplePointsInGroup(float param, unsigned igroup, GeometryArray * geos)
{
    Vector3F sum = Vector3F::Zero;
    unsigned i = 0;
    for(; i< N(); i++) {
        if(group(i) == igroup) {
            BaseCurve * c = static_cast<BaseCurve *>(geos->geometry(i));
            sum += c->interpolate(param);
        }
    }
    sum *= 1.f / (float)countPerGroup(igroup);
    return sum;
}

