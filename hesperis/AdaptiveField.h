#pragma once
#include "AdaptiveGrid.h"
#include <AField.h>
#include <Geometry.h>
class ATetrahedronMesh;
class AdaptiveField : public AdaptiveGrid, public AField
{
public:
    AdaptiveField();
	virtual ~AdaptiveField();
	
	virtual FieldType fieldType() const;
    
    virtual void create(KdIntersection * tree,
						ATetrahedronMesh * mesh,
						int maxLevel = 7);
    
    void addFloatChannel(const std::string & name);
	void addVec3Channel(const std::string & name);
	
    void computeChannelValue(const std::string & channelName,
                             TypedBuffer * source,
                             BaseSampler * sampler);
    
    virtual void verbose() const;
protected:
    
private:
    void createSamples(KdIntersection * tree,
                       ATetrahedronMesh * mesh);
    
    void setSampleParam(unsigned code,
                        Geometry::ClosestToPointTestResult * ctx,
                        ATetrahedronMesh * mesh);
    
	void findNeighbours();
    
    void sampleAChannel(TypedBuffer * chan,
                                   TypedBuffer * source,
                                   BaseSampler * sampler);
	
	template<typename T, typename Ts>
    void sampleChannelValue(TypedBuffer * chan,
                            TypedBuffer * source, 
                            BaseSampler * sampler)
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
        T * io = chan->typedData<T>();
        unsigned idx;
        sdb::CellHash * c = cells();
        c->begin();
        while(!c->end()) {
            if(c->value()->visited < 1) {
                idx = c->value()->index;
                
                CellNeighbourInds * nei = m_neighbours->find(c->key());
                io[idx] = interpolateValue(io, nei);
            }
            c->next();
        }
    }
    
    template<typename T>
    T interpolateValue(T * io, CellNeighbourInds * nei)
    {
        T r;
        int nnei = 0;
        int i = 0;
        bool first = true;
        for(;i<6;i++) {
            if(nei->hasSide(i)) {
                if(first) {
                    r = interpolateValue(io, nei->side(i));
                    first = false;
                }
                else
                    r += interpolateValue(io, nei->side(i));
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
	void checkNeighbours();
    
private:
    struct SampleParam {
        unsigned _vertices[4];
        float _contributes[4];
    };
    
typedef sdb::Array<unsigned, SampleParam> SampleHash;
    SampleHash * m_sampleParams;
typedef sdb::Array<unsigned, CellNeighbourInds> NeighbourHash;	
    NeighbourHash * m_neighbours;
};
