#include "AdaptiveGrid.h"
#include <AField.h>
#include <Geometry.h>
class ATetrahedronMesh;
class AdaptiveField : public AdaptiveGrid, public AField
{
public:
    AdaptiveField(const BoundingBox & bound);
    virtual ~AdaptiveField();
    
    virtual void create(KdIntersection * tree, AField * source,
						ATetrahedronMesh * mesh,
						SamplerType st,
						int maxLevel = 6);
protected:

private:
	void setCellValues(KdIntersection * tree, 
					AField * source,
					ATetrahedronMesh * mesh,
					SamplerType st);
	void setACellValues(unsigned idata, 
					Geometry::ClosestToPointTestResult * ctx,
					AField * source,
					ATetrahedronMesh * mesh,
					SamplerType st,
					const std::vector<std::string > & channelNames);
};
