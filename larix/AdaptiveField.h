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
						int maxLevel = 6);
protected:

private:
	void setCellValues(KdIntersection * tree, 
					AField * source,
					ATetrahedronMesh * mesh);
    
	void setACellValues(unsigned idata, 
                        BaseSampler * sampler,
					AField * source,
					const std::vector<std::string > & channelNames);
    
    void createSamples(KdIntersection * tree,
                       ATetrahedronMesh * mesh);
    
    void setSampleParam(unsigned code,
                        Geometry::ClosestToPointTestResult * ctx,
                        ATetrahedronMesh * mesh);
    
    void sampleCellValues(AField * source, BaseSampler * sampler);
private:
    struct SampleParam {
        unsigned _vertices[4];
        float _contributes[4];
    };
    
typedef sdb::Array<unsigned, SampleParam> SampleHash;
    SampleHash * m_sampleParams;
};
