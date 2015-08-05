#include "AdaptiveField.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
#include <ATetrahedronMesh.h>

AdaptiveField::AdaptiveField(const BoundingBox & bound) :
    AdaptiveGrid(bound)
{
    m_sampleParams = new SampleHash;
	m_neighbours = new NeighborHash;
}

AdaptiveField::~AdaptiveField() 
{
    delete m_sampleParams;
	delete m_neighbours;
}

void AdaptiveField::create(KdIntersection * tree, AField * source,
					ATetrahedronMesh * mesh,
					int maxLevel)
{
	AdaptiveGrid::create(tree, maxLevel);	
	setCellValues(tree, source, mesh);
}

void AdaptiveField::setCellValues(KdIntersection * tree, 
					AField * source,
					ATetrahedronMesh * mesh)
{
    createSamples(tree, mesh);
    sampleCellValues(source, mesh->sampler());
	findNeighbours();
}

void AdaptiveField::createSamples(KdIntersection * tree,
                       ATetrahedronMesh * mesh)
{
    m_sampleParams->clear();
	
	Geometry::ClosestToPointTestResult cls;
	std::vector<unsigned> elms;
	unsigned i = 0;
	BoundingBox box;
	sdb::CellHash * c = cells();
	c->begin();
	while(!c->end()) {
		box = cellBox(c->key(), c->value()->level);
		gjk::IntersectTest::SetA(box);
		tree->countElementIntersectBox(elms, box);
		if(elms.size() > 0) {
			c->value()->visited = 1;
			cls.reset(cellCenter(c->key()), 1e8f);
			mesh->closestToPointElms(elms, &cls);

            setSampleParam(c->key(), &cls, mesh);
		}
		else
			c->value()->visited = 0;	

        c->value()->index = i;	
		c->next();
		i++;
	}
    std::cout<<"\n n samples "<<m_sampleParams->size();
}

void AdaptiveField::setSampleParam(unsigned code,
                        Geometry::ClosestToPointTestResult * ctx,
                        ATetrahedronMesh * mesh)
{
    SampleParam * v = new SampleParam;
    unsigned * tet = mesh->tetrahedronIndices(ctx->_icomponent);
    v->_vertices[0] = tet[0];
    v->_vertices[1] = tet[1];
    v->_vertices[2] = tet[2];
    v->_vertices[3] = tet[3];
    v->_contributes[0] = ctx->_contributes[0];
    v->_contributes[1] = ctx->_contributes[1];
    v->_contributes[2] = ctx->_contributes[2];
    v->_contributes[3] = ctx->_contributes[3];
    m_sampleParams->insert(code, v);
}

void AdaptiveField::sampleCellValues(AField * source, BaseSampler * sampler)
{
    const unsigned n = numCells();
	std::vector<std::string > names;
	source->getChannelNames(names);
	std::vector<std::string >::const_iterator it = names.begin();
	for(;it!=names.end();++it) {
		source->useChannel(*it);
		if(source->currentChannel()->valueType() == TypedBuffer::TFlt)
			addFloatChannel(*it, n);
		else if(source->currentChannel()->valueType() == TypedBuffer::TVec3)
			addVec3Channel(*it, n);
	}
    if(numChannels() < 1) {
        std::cout<<"\n field has no channels";
        return;
    }
    
    sdb::CellHash * c = cells();
    c->begin();
	while(!c->end()) {
		if(c->value()->visited) {
            SampleParam * param = m_sampleParams->find(c->key());
            sampler->set(param->_vertices, param->_contributes);
			setACellValues(c->value()->index,
                           sampler,
                           source,
                           names);
		}
		c->next();
	}
}

void AdaptiveField::setACellValues(unsigned idata, 
                                   BaseSampler * sampler,
					AField * source,
					const std::vector<std::string > & channelNames)
{
	std::vector<std::string >::const_iterator it = channelNames.begin();
	for(;it!=channelNames.end();++it) {
		TypedBuffer * chan = namedChannel(*it);
        source->useChannel(*it);
		if(chan->valueType() == TypedBuffer::TFlt) {
			float * dst = chan->typedData<float>();
            dst[idata] = source->sample<float, TetrahedronSampler>(reinterpret_cast<TetrahedronSampler *>(sampler));
		}
		else if(chan->valueType() == TypedBuffer::TVec3) {
			Vector3F * dst = chan->typedData<Vector3F>();
            dst[idata] = source->sample<Vector3F, TetrahedronSampler>(reinterpret_cast<TetrahedronSampler *>(sampler));
		}
	}
}

void AdaptiveField::findNeighbours()
{
	sdb::CellHash * c = cells();
    c->begin();
	while(!c->end()) {
		m_neighbours->insert(c->key(), findNeighbourCells(c->key()));
		c->next();
	}
	std::cout<<" neighbour hash size "<<m_neighbours->size();
}
//:~