#include "BccLattice.h"
#include <GeoDrawer.h>
#include "bcc_common.h"
BccLattice::BccLattice(const BoundingBox & bound) :
    CartesianGrid(bound)
{   
    m_greenEdges = new sdb::EdgeHash;
}

BccLattice::~BccLattice() {}

void BccLattice::add14Node(const Vector3F & center, float h)
{
	const unsigned ccenter = mortonEncode(center);
	unsigned cgreen;
	Vector3F corner;
    int i;
    float hh = h * .5f;
	for(i=0; i < 8; i++) {
        corner = center + Vector3F(hh * OctChildOffset[i][0], 
        hh * OctChildOffset[i][1], 
        hh * OctChildOffset[i][2]);
        
        addGrid(corner);
    }
    for(i=0; i < 6; i++) {
        corner = center + Vector3F(h * HexHeighborOffset[i][0], 
        h * HexHeighborOffset[i][1], 
        h * HexHeighborOffset[i][2]);
        
        cgreen = addGrid(corner);
        m_greenEdges->addEdge(ccenter, cgreen);
    }
    addGrid(center);
}

void BccLattice::connect24Tetrahedron(const Vector3F & center, float h)
{
    const unsigned ccenter = mortonEncode(center);
    unsigned cgreen;
	Vector3F corner;
	int i;
	for(i=0; i < 6; i++) {
        corner = center + Vector3F(h * HexHeighborOffset[i][0], 
        h * HexHeighborOffset[i][1], 
        h * HexHeighborOffset[i][2]);
        
        cgreen = mortonEncode(corner);
        
		sdb::EdgeValue * edge = m_greenEdges->findEdge(ccenter, cgreen);
		if(!edge) {
		    std::cout<<" edge "<<i<<" connected to "<<center<<" doesn't exist!\n";
		    continue;
		}
		if(edge->visited == 0) {
		      
		    edge->visited = 1;
		}
    }
}

const unsigned BccLattice::numGreenEdges() const
{
	return m_greenEdges->size();
}

void BccLattice::draw(GeoDrawer * drawer)
{
    sdb::CellHash * latticeNode = cells();
	drawer->setColor(0.f, 0.f, 0.3f);
	float h = cellSizeAtLevel(8);
	Vector3F l;
	latticeNode->begin();
	while(!latticeNode->end()) {
	    l = gridOrigin(latticeNode->key());
	    drawer->cube(l, h);
	    latticeNode->next();
	}
	drawGreenEdges();
}

void BccLattice::drawGreenEdges()
{
	unsigned a, b;
	Vector3F pa, pb;
	glColor3f(0.f, 1.f, 0.f);
	glBegin(GL_LINES);
	m_greenEdges->begin();
	while(!m_greenEdges->end()) {
		m_greenEdges->connectedTo(a, b);
		pa = gridOrigin(a);
		pb = gridOrigin(b);
		
		glVertex3fv((GLfloat *)&pa);
		glVertex3fv((GLfloat *)&pb);
		m_greenEdges->next();
	}
	glEnd();
}
