#include "HarmonicCoord.h"
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"

HarmonicCoord::HarmonicCoord() {}
HarmonicCoord::~HarmonicCoord() {}

void HarmonicCoord::setMesh(BaseMesh * mesh)
{
	BaseField::setMesh(mesh);
	
	printf("init laplace deformer");
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
	m_topology = msh->connectivity();
	initialCondition();
}

void HarmonicCoord::precompute(std::vector<WeightHandle *> & anchors)
{
	m_anchors = anchors;
	
	m_numAnchors = 0;
	for(std::vector<WeightHandle *>::iterator it = anchors.begin(); it != anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			m_numAnchors++;
		}
	}
	
	int neighborIdx;
	LaplaceMatrixType L(m_numVertices + m_numAnchors, m_numVertices);
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];

		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			L.insert(i, neighborIdx) = -neighbor->weight;
		}
		L.insert(i, i) = 1.0f;
	}
	
	int irow = m_numVertices;
	for(std::vector<WeightHandle *>::iterator it = anchors.begin(); it != anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			L.coeffRef(irow, idx) = 1.f;
			irow++;
		}
	}
	
	m_LT = L.transpose();
	LaplaceMatrixType m_M = m_LT * L;
	m_llt.compute(m_M);
}

unsigned HarmonicCoord::numAnchorPoints() const
{
	return m_numAnchors;
}

void HarmonicCoord::initialCondition()
{
	int ne = 0;
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];
		ne += adj.getNumNeighbors();
	}
	printf("\nedge count: %i\n", ne);
}

void HarmonicCoord::prestep()
{
	m_b.resize(m_numVertices + m_numAnchors);
	m_b.setZero();
	int irow = (int)m_numVertices;

	for(std::vector<WeightHandle *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			m_b(irow) = ap->w;
			irow++;
		}
	}
	
	m_b = m_LT * m_b;
}

char HarmonicCoord::solve()
{
	prestep();
	Eigen::VectorXf x = m_llt.solve(m_b);
	
	float r;
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		r = x(i);
		if(r < 0.f) r = 0.f;
		if(r > 1.f) r = 1.f;
		m_value[i] = r;
	}
	plotColor();
	return 1;
}

bool HarmonicCoord::allZero() const
{
	for(std::vector<WeightHandle *>::const_iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			if(ap->w > 10e-3)
				return false;
		}
	}
	return true;
}

unsigned HarmonicCoord::genNonZeroIndices(std::vector<unsigned > & dst) const
{
	if(allZero()) return 0;
	for(unsigned i = 0; i < m_numVertices; i++) {
		//if(isAnchorPoint(i))
		//	dst.push_back(i);
		//else //if(m_value[i] > 10e-3)
			dst.push_back(i);
	}
	return (unsigned)dst.size();
}

bool HarmonicCoord::isAnchorPoint(unsigned i) const
{
	for(std::vector<WeightHandle *>::const_iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		unsigned idx;
		(*it)->firstPoint(idx);  
		if(i == idx) return true;
	}
	return false;
}

void HarmonicCoord::genAnchorIndices(std::vector<unsigned > & dst) const
{
	for(std::vector<WeightHandle *>::const_iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			dst.push_back(idx);
		}
	}
}

bool HarmonicCoord::hasNoEffect() const
{
	return allZero();
}
