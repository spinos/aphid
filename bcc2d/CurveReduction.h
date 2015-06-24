#include "SampleGroup.h"
#include <QuickSort.h>

class BezierCurve;
class Geometry;
class GeometryArray;
class CurveReduction : public KMeanSampleGroup {
public:
    CurveReduction();
	virtual ~CurveReduction();
	GeometryArray * compute(GeometryArray * curves, float distanceThreshold);
	
protected:
// override KMeansClustering
	virtual void initialGuess(const Vector3F * pos);
    
private:
    GeometryArray * mergeCurves(GeometryArray * curves);
    unsigned minimumNumCvsInGroup(unsigned igroup, GeometryArray * curves) const;
    BezierCurve * duplicateCurve(unsigned igroup, GeometryArray * curves);
    BezierCurve * mergeCurvesInGroup(unsigned igroup, GeometryArray * geos);
    Vector3F samplePointsInGroup(float param, unsigned igroup, GeometryArray * geos);
	typedef QuickSortPair<float, unsigned > GapInd;
	void computeSampleGaps(Vector3F * samples, unsigned n, GeometryArray * curves);
	int computeM(unsigned n, float distanceThreshold);
private:
	GapInd * m_gapHash;
};
