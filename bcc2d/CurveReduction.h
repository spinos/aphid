#include "SampleGroup.h"
class BezierCurve;
class Geometry;
class GeometryArray;
class CurveReduction : public KMeanSampleGroup {
public:
    CurveReduction();
	virtual ~CurveReduction();
    
    GeometryArray * compute(GeometryArray * curves, float alpha);
protected:

private:
    GeometryArray * mergeCurves(GeometryArray * curves);
    unsigned minimumNumCvsInGroup(unsigned igroup, GeometryArray * curves) const;
    BezierCurve * duplicateCurve(unsigned igroup, GeometryArray * curves);
    BezierCurve * mergeCurvesInGroup(unsigned igroup, GeometryArray * geos);
    Vector3F samplePointsInGroup(float param, unsigned igroup, GeometryArray * geos);
private:
};
