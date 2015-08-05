#include "AdaptiveField.h"
#include <KdIntersection.h>
#include <GjkIntersection.h>
#include <ATetrahedronMesh.h>

AdaptiveField::AdaptiveField(const BoundingBox & bound) :
    AdaptiveGrid(bound)
{
    m_sampleParams = new SampleHash;
	m_neighbours = new BaseBuffer;
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
    int i=0;
    for(;i<24;i++) interpolate();
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
        setChannelZero(*it);
	}
    if(numChannels() < 1) {
        std::cout<<"\n field has no channels";
        return;
    }
    
    it = names.begin();
	for(;it!=names.end();++it)
        sampleNamedChannel(*it, source, sampler);
}

void AdaptiveField::sampleNamedChannel(const std::string & channelName,
                                       AField * source, BaseSampler * sampler)
{
    source->useChannel(channelName);
    TypedBuffer * chan = namedChannel(channelName);
    
    if(chan->valueType() == TypedBuffer::TFlt)
        sampleChannelValue<float, TetrahedronSampler>(chan, source, sampler);
    else if(chan->valueType() == TypedBuffer::TVec3)
        sampleChannelValue<Vector3F, TetrahedronSampler>(chan, source, sampler);
}

void AdaptiveField::findNeighbours()
{
    m_neighbours->create(numCells() * 24 * 4);
    CellNeighbourInds * neis = neighbours();

	sdb::CellHash * c = cells();
    c->begin();
	while(!c->end()) {
		findNeighbourCells( neis, c->key(), c->value() );
		c->next();
        neis++;
	}
}

AdaptiveField::CellNeighbourInds * AdaptiveField::neighbours() const
{ return (CellNeighbourInds *)m_neighbours->data(); }

void AdaptiveField::interpolate()
{
    std::vector<std::string > names;
	getChannelNames(names);
	std::vector<std::string >::const_iterator it = names.begin();
	for(;it!=names.end();++it) {
		TypedBuffer * chan = namedChannel(*it);
        if(chan->valueType() == TypedBuffer::TFlt)
            interpolateChannel<float>(chan);
        else if(chan->valueType() == TypedBuffer::TVec3)
            interpolateChannel<Vector3F>(chan);
	}
}
//:~