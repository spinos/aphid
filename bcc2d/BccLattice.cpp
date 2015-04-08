#include "BccLattice.h"
#include <GeoDrawer.h>
#include "bcc_common.h"
BccLattice::BccLattice(const BoundingBox & bound) :
    CartesianGrid(bound)
{   
    m_greenEdges = new sdb::EdgeHash;
    m_tetrahedrons = 0;
    m_numTetrahedrons = 0;
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

void BccLattice::prepareTetrahedron()
{
    const unsigned noctahedron = m_greenEdges->size();
    m_tetrahedrons = new Tetrahedron[noctahedron * 4];
    m_numTetrahedrons = 0;
}

void BccLattice::connect24Tetrahedron(const Vector3F & center, float h)
{
    const unsigned ccenter = mortonEncode(center);
    unsigned cgreen;
	Vector3F corner;
	unsigned vOctahedron[6];
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
		    encodeOctahedronVertices(center, h, i, vOctahedron);
		    add4Tetrahedrons(vOctahedron);
		    edge->visited = 1;
		}
    }
}

const unsigned BccLattice::numGreenEdges() const
{ return m_greenEdges->size(); }

const unsigned BccLattice::numTetrahedrons() const
{ return m_numTetrahedrons; }

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
	// drawGreenEdges();
	drawer->setWired(1);
	drawTetrahedrons();
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

void BccLattice::drawTetrahedrons()
{
    glColor3f(0.f, 0.1f, 0.2f);
    glBegin(GL_TRIANGLES);
    unsigned i, j;
    Vector3F q;
    for(i=0; i< m_numTetrahedrons; i++) {
        Tetrahedron * tet = &m_tetrahedrons[i];
        for(j=0; j< 12; j++) {
            q = gridOrigin(tet->v[TetrahedronToTriangleVertex[j]]);
            glVertex3fv((GLfloat *)&q);
        }
    }
    glEnd();
}

void BccLattice::encodeOctahedronVertices(const Vector3F & q, float h, int offset, unsigned * v) const
{
    Vector3F corner;
    int i;
    for(i=0; i<6; i++) {
        corner = q + Vector3F(h * HexOctahedronOffset[offset][i][0],
                              h * HexOctahedronOffset[offset][i][1],
                              h * HexOctahedronOffset[offset][i][2]);
        
        v[i] = mortonEncode(corner);
        // if(!findGrid(v[i])) std::cout<<" cannot find grid "<<corner<<" ";
    }
}

void BccLattice::add4Tetrahedrons(unsigned * vOctahedron)
{
    int i;
    for(i=0; i<4; i++) {
        Tetrahedron * tet = &m_tetrahedrons[m_numTetrahedrons];
        tet->v[0] = vOctahedron[OctahedronToTetrahedronVetex[i][0]];
        tet->v[1] = vOctahedron[OctahedronToTetrahedronVetex[i][1]];
        tet->v[2] = vOctahedron[OctahedronToTetrahedronVetex[i][2]];
        tet->v[3] = vOctahedron[OctahedronToTetrahedronVetex[i][3]];
        m_numTetrahedrons++;
    }
}

