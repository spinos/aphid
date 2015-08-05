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
    CellNeighbourInds * neighbours() const;
private:
	void setCellValues(KdIntersection * tree, 
					AField * source,
					ATetrahedronMesh * mesh);
    
    void createSamples(KdIntersection * tree,
                       ATetrahedronMesh * mesh);
    
    void setSampleParam(unsigned code,
                        Geometry::ClosestToPointTestResult * ctx,
                        ATetrahedronMesh * mesh);
    
    void sampleCellValues(AField * source, BaseSampler * sampler);
	
	void findNeighbours();
    
    void sampleNamedChannel(const std::string & channelName,
                            AField * source, BaseSampler * sampler);
	
	template<typename T, typename Ts>
    void sampleChannelValue(TypedBuffer * chan,
                            AField * source, BaseSampler * sampler)
    {
        T * dst = chan->typedData<T>();
        
        sdb::CellHash * c = cells();
        c->begin();
        while(!c->end()) {
            if(c->value()->visited) {
                SampleParam * param = m_sampleParams->find(c->key());
                sampler->set(param->_vertices, param->_contributes);
                dst[c->value()->index] = source->sample<T, Ts>(reinterpret_cast<Ts *>(sampler));
            }
            c->next();
        }
    }
    
    template<typename T>
    void interpolateChannel(TypedBuffer * chan)
    {
        CellNeighbourInds * nei = neighbours();
        T * io = chan->typedData<T>();
        unsigned idx;
        sdb::CellHash * c = cells();
        c->begin();
        while(!c->end()) {
            if(c->value()->visited < 1) {
                idx = c->value()->index;
                io[idx] = interpolateValue(io, nei[idx]);
            }
            c->next();
        }
    }
    
    template<typename T>
    T interpolateValue(T * io, CellNeighbourInds & nei)
    {
        T r;
        int nnei = 0;
        int i = 0;
        bool first = true;
        for(;i<6;i++) {
            if(nei.hasSide(i)) {
                if(first) {
                    r = interpolateValue(io, nei.side(i));
                    first = false;
                }
                else
                    r += interpolateValue(io, nei.side(i));
                nnei++;
            }
        }
        
        if(nnei > 1) r *= 1.f / (float)nnei;
        // else std::cout<<"zero neighbour";
        return r;
    }
    
    template<typename T>
    T interpolateValue(T * io, unsigned * idx)
    {
        bool first = true;
        T r;
        int n = 0;
        int i = 0;
        for(;i<4;i++) {
            if( CellNeighbourInds::IsValidIndex(idx[i]) ) {
                if(first) {
                    r = io[idx[i]];
                    first = false;
                }
                else
                    r += io[idx[i]];
                n++;
            }
        }
        if(n > 1) r *= 1.f / (float)n;
        return r;
    }
    
    void interpolate();
    
private:
    struct SampleParam {
        unsigned _vertices[4];
        float _contributes[4];
    };
    
typedef sdb::Array<unsigned, SampleParam> SampleHash;
    SampleHash * m_sampleParams;
typedef sdb::Array<unsigned, CellNeighbourInds> NeighbourHash;	
    BaseBuffer * m_neighbours;
};
