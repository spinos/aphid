#include "HarmonicCoord.h"
#include "VertexAdjacency.h"
#include "MeshLaplacian.h"
#include "ControlGraph.h"
#include <Anchor.h>
HarmonicCoord::HarmonicCoord() 
{
	m_constrainValues = 0;
}

HarmonicCoord::~HarmonicCoord() 
{
	if(m_constrainValues) delete[] m_constrainValues;
}

void HarmonicCoord::setMesh(BaseMesh * mesh)
{
	BaseField::setMesh(mesh);
	
	printf("init laplace deformer");
	MeshLaplacian * msh = static_cast <MeshLaplacian *>(m_mesh);
	m_topology = msh->connectivity();
	initialCondition();
}

void HarmonicCoord::precompute(std::vector<Anchor *> & anchors)
{
	m_anchors = anchors;
	
	m_numAnchors = 0;
	for(std::vector<Anchor *>::iterator it = anchors.begin(); it != anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			m_numAnchors++;
		}
	}
	
	m_constrainValues = new float[m_numAnchors];
	
	int neighborIdx;
	LaplaceMatrixType L(m_numVertices + m_numAnchors, m_numVertices);
	for(int i = 0; i < (int)m_numVertices; i++) {
		VertexAdjacency & adj = m_topology[i];

		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighborOrderedByVertexIdx(); !adj.isLastNeighborOrderedByVertexIdx(); neighbor = adj.nextNeighborOrderedByVertexIdx()) {
			neighborIdx = neighbor->v->getIndex();
			L.insert(i, neighborIdx) = -neighbor->weight;
		}
		L.insert(i, i) = 1.0;
	}
	
	int irow = m_numVertices;
	for(std::vector<Anchor *>::iterator it = anchors.begin(); it != anchors.end(); ++it) {
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			L.coeffRef(irow, idx) = 1.0;
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

	unsigned ianchor = 0;
	for(std::vector<Anchor *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		float wei = m_constrainValues[ianchor];
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			m_b(irow) = wei;
			irow++;
		}
		ianchor++;
	}
	m_b = m_LT * m_b;
}

char HarmonicCoord::solve(unsigned iset)
{
	m_activeValue = iset;
	prestep();
	Eigen::VectorXd x = m_llt.solve(m_b);
	
	float r;
	float *v = value(iset);
	for(int i = 0; i < (int)m_numVertices; i++) {
		r = x(i);
		if(r < 0.f) r = 0.f;
		if(r > 1.f) r = 1.f;
		v[i] = r;
	}
	
	checkConstrain(iset);
	return 1;
}

void HarmonicCoord::setConstrain(unsigned idx, float val)
{
	m_constrainValues[idx] = val;
}

void HarmonicCoord::checkConstrain(unsigned iset)
{
	float *v = value(iset);
	unsigned ianchor = 0;
	for(std::vector<Anchor *>::iterator it = m_anchors.begin(); it != m_anchors.end(); ++it) {
		float c = m_constrainValues[ianchor];
		unsigned idx;
		for(Anchor::AnchorPoint * ap = (*it)->firstPoint(idx); (*it)->hasPoint(); ap = (*it)->nextPoint(idx)) {
			printf("c %f v %f \n", c, v[idx]);
		}
		ianchor++;
	}
}

