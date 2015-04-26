#include "BccLattice.h"
#include <GeoDrawer.h>
#include <BezierCurve.h>
#include <BaseLog.h>
#include <boost/format.hpp>
#include "bcc_common.h"
#include <KdIntersection.h>
Vector3F BccLattice::NodeCenterOffset;

BccLattice::BccLattice(const BoundingBox & bound) :
    CartesianGrid(bound)
{   
    m_greenEdges = new sdb::EdgeHash;
    m_tetrahedrons = 0;
    m_numTetrahedrons = 0;
}

BccLattice::~BccLattice() {}

const Vector3F BccLattice::nodeCenter(unsigned code) const
{ return gridOrigin(code) + NodeCenterOffset; }

void BccLattice::add38Node(const Vector3F & center, float h)
{
	const unsigned ccenter = mortonEncode(center);
	unsigned cgreen, cgreen1;
	Vector3F corner, corner1;
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
    for(i=0; i < 24; i++) {
        corner = center + Vector3F(h * ParallelEdgeOffset[i][0][0], 
        h * ParallelEdgeOffset[i][0][1], 
        h * ParallelEdgeOffset[i][0][2]);
        
        cgreen = addGrid(corner);
        
        corner1 = center + Vector3F(h * ParallelEdgeOffset[i][1][0], 
        h * ParallelEdgeOffset[i][1][1], 
        h * ParallelEdgeOffset[i][1][2]);
        
        cgreen1 = addGrid(corner1);
        
        m_greenEdges->addEdge(cgreen, cgreen1);
    }
    addGrid(center);
}

void BccLattice::prepareTetrahedron()
{
    const float hh = cellSizeAtLevel(11);
    NodeCenterOffset.set(hh, hh, hh);
    const unsigned noctahedron = m_greenEdges->size();
    m_tetrahedrons = new Tetrahedron[noctahedron * 4];
    m_numTetrahedrons = 0;
}

void BccLattice::touchIntersectedTetrahedron(const Vector3F & center, float h,
												KdIntersection * tree)
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
		    touch4Tetrahedrons(vOctahedron, tree);
		    edge->visited = 1;
		}
    }
}

void BccLattice::add24Tetrahedron(const Vector3F & center, float h)
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
		    // std::cout<<" edge "<<i<<" connected to "<<center<<" doesn't exist!\n";
		    continue;
		}
		if(edge->visited == 0) {
		    encodeOctahedronVertices(center, h, i, vOctahedron);
		    addTetrahedronsAllNodeVisited(vOctahedron);
		    edge->visited = 1;
		}
    }
}

void BccLattice::addNeighborTetrahedron(const Vector3F & center, float h)
{
    Vector3F corner;
    int i;
	for(i=0; i < 6; i++) {
        corner = center + Vector3F(h * HexHeighborOffset[i][0], 
        h * HexHeighborOffset[i][1], 
        h * HexHeighborOffset[i][2]);
        
        add24Tetrahedron(corner, h);
    }
}

const unsigned BccLattice::numGreenEdges() const
{ return m_greenEdges->size(); }

const unsigned BccLattice::numTetrahedrons() const
{ return m_numTetrahedrons; }

const unsigned BccLattice::numVertices() const
{ return m_visitedNodes; }

void BccLattice::draw(GeoDrawer * drawer, unsigned * anchored)
{/*
    drawer->setWired(0);
	glColor3f(0.3f, 0.4f, 0.33f);
	drawTetrahedrons();
	drawer->setWired(1);
	glColor3f(.03f, .14f, .44f);
	drawTetrahedrons();*/
	glColor3f(.03f, .14f, .44f);
	drawVisitedNodes(drawer);
}

void BccLattice::drawAllNodes(GeoDrawer * drawer)
{
    drawer->setWired(0);
    sdb::CellHash * latticeNode = cells();
	float h = cellSizeAtLevel(11);
	Vector3F l;
	latticeNode->begin();
	while(!latticeNode->end()) {
	    if(latticeNode->value()->visited)
	        glColor3f(0.1f, 0.2f, 0.5f);
	    else 
			glColor3f(0.3f, 0.3f, 0.3f);
	    
	    l = nodeCenter(latticeNode->key());
	    drawer->cube(l, h);
	    
	    latticeNode->next();
	}
}

void BccLattice::drawVisitedNodes(GeoDrawer * drawer)
{
    drawer->setWired(0);
	glColor3f(0.1f, 0.2f, 0.5f);
    sdb::CellHash * latticeNode = cells();
	float h = cellSizeAtLevel(11);
	Vector3F l;
	latticeNode->begin();
	while(!latticeNode->end()) {
	    if(latticeNode->value()->visited) {
			l = nodeCenter(latticeNode->key());
			drawer->cube(l, h);
	    }
	    latticeNode->next();
	}
}

void BccLattice::drawGreenEdges()
{
	unsigned a, b;
	Vector3F pa, pb;
	glBegin(GL_LINES);
	m_greenEdges->begin();
	while(!m_greenEdges->end()) {
	    if(m_greenEdges->value()->visited) glColor3f(0.f, .8f, 0.f);
	    else glColor3f(0.f, .5f, 0.f);
	    
		m_greenEdges->connectedTo(a, b);
		pa = nodeCenter(a);
		pb = nodeCenter(b);
		
		glVertex3fv((GLfloat *)&pa);
		glVertex3fv((GLfloat *)&pb);
		m_greenEdges->next();
	}
	glEnd();
}

void BccLattice::drawTetrahedrons()
{
    glBegin(GL_TRIANGLES);
    unsigned i, j;
    Vector3F q;
    for(i=0; i< m_numTetrahedrons; i++) {
        Tetrahedron * tet = &m_tetrahedrons[i];
        for(j=0; j< 12; j++) {
            q = nodeCenter(tet->v[TetrahedronToTriangleVertex[j]]);
            glVertex3fv((GLfloat *)&q);
        }
    }
    glEnd();
}	

void BccLattice::drawTetrahedrons(unsigned * anchored)
{
    glBegin(GL_TRIANGLES);
    unsigned i, j;
    Vector3F q;
	unsigned a[4];
    for(i=0; i< m_numTetrahedrons; i++) {
        Tetrahedron * tet = &m_tetrahedrons[i];
		for(j=0; j<4; j++) {
			sdb::CellValue * found = findGrid(tet->v[j]);
			a[j] = anchored[found->index];
		}
		
        for(j=0; j< 12; j++) {
            q = nodeCenter(tet->v[TetrahedronToTriangleVertex[j]]);
			
			if(a[TetrahedronToTriangleVertex[j]])
				glColor3f(.993f, .14f, .04f);
			else
				glColor3f(.03f, .14f, .44f);
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

bool BccLattice::isCurveClosetToTetrahedron(const Vector3F * p, BezierCurve * curve) const
{
    int i;
    Vector3F q;
    for(i=0; i<4; i++) {
        if(curve->distanceToPoint(p[i], q) < 0.2f) return true;
    }
    return curve->intersectTetrahedron(p);
}

void BccLattice::touch4Tetrahedrons(unsigned * vOctahedron, KdIntersection * tree)
{
    unsigned code[4];
    Vector3F tet[4];
    int i, j;
    for(i=0; i<4; i++) {
        for(j=0; j<4; j++) {
            code[j] = vOctahedron[OctahedronToTetrahedronVetex[i][j]];
            tet[j] = nodeCenter(code[j]);
        }
        if(tree->intersectTetrahedron(tet)) {
            for(j=0; j<4; j++) {
                code[j] = vOctahedron[OctahedronToTetrahedronVetex[i][j]];
                sdb::CellValue * found = findGrid(code[j]);
                if(!found) {
                    std::cout<<" cannot find grid "<<nodeCenter(code[j])<<" ";
                    break;
                }
                found->visited = 1;
            }
        }
    }
}

void BccLattice::addTetrahedronsAllNodeVisited(unsigned * vOctahedron)
{
    unsigned code;
    bool allVisited;
    int i, j;
    for(i=0; i<4; i++) {
        allVisited = 1;
        for(j=0; j<4; j++) {
            code = vOctahedron[OctahedronToTetrahedronVetex[i][j]];
            sdb::CellValue * found = findGrid(code);
            if(!found) {
                // std::cout<<" cannot find grid "<<nodeCenter(code)<<" ";
                allVisited = 0;
                continue;
            }
            if(!found->visited) allVisited = 0;
        }
        
        if(allVisited) {
            Tetrahedron * t = &m_tetrahedrons[m_numTetrahedrons];
            for(j=0; j<4; j++) {
                code = vOctahedron[OctahedronToTetrahedronVetex[i][j]];
                t->v[j] = code;
            }
            m_numTetrahedrons++;
        }
    }
}

void BccLattice::untouchGreenEdges()
{
    m_greenEdges->begin();
	while(!m_greenEdges->end()) {
	    sdb::EdgeValue * edge = m_greenEdges->value();
		edge->visited = 0;
		m_greenEdges->next();
	}
	glEnd();
}

void BccLattice::countVisitedNodes()
{
    m_visitedNodes = 0;
    sdb::CellHash * latticeNode = cells();
    latticeNode->begin();
	while(!latticeNode->end()) {
	    if(latticeNode->value()->visited) {
	        latticeNode->value()->index = m_visitedNodes;
	        m_visitedNodes++;
	    }
	    latticeNode->next();
	}
}

void BccLattice::logTetrahedronMesh()
{
    Vector3F p;
    BaseLog log("./tetmesh.txt");
    
    log.write(boost::str(boost::format("static const unsigned TetraNumVertices = %1%;\n") % numVertices()));
	log.write(boost::str(boost::format("static const float TetraP[%1%][3] = {\n") % numVertices()));
	sdb::CellHash * latticeNode = cells();
    
	latticeNode->begin();
	while(!latticeNode->end()) {
	    if(latticeNode->value()->visited) {
	        p = nodeCenter(latticeNode->key());
	        log.write(boost::str(boost::format("{%1%f,%2%f,%3%f}") % p.x % p.y % p.z));
	        if(latticeNode->value()->index < numVertices()-1) log.write(",\n");
	        else log.write("\n");
	    }
	    latticeNode->next();
	}
	log.write("};\n");
	log.write(boost::str(boost::format("static const unsigned TetraNumTetrahedrons = %1%;\n") % numTetrahedrons()));
	log.write(boost::str(boost::format("static const unsigned TetraIndices[%1%][4] = {\n") % numTetrahedrons()));
	
	unsigned i, j;
	unsigned v[4];
	for(i=0; i< numTetrahedrons(); i++) {
        Tetrahedron * tet = &m_tetrahedrons[i];
        for(j=0; j< 4; j++) {
            sdb::CellValue * found = findGrid(tet->v[j]);
            v[j] = found->index;
        }
        log.write(boost::str(boost::format("{%1%,%2%,%3%,%4%}") % v[0] % v[1] % v[2] % v[3]));
	    if(i < numTetrahedrons()-1) log.write(",\n");
	    else log.write("\n");
    }
    
	log.write("};\n");
}

bool BccLattice::intersectTetrahedron(const Vector3F * tet, BezierSpline * splines, unsigned numSplines) const
{
	unsigned i = 0;
	for(; i<numSplines; i++) {
		BoundingBox tbox;
		tbox.expandBy(tet[0]);
		tbox.expandBy(tet[1]);
		tbox.expandBy(tet[2]);
		tbox.expandBy(tet[3]);
		if(BezierCurve::intersectTetrahedron(splines[i], tet, tbox))
			return true;
	}
	return false;
}

void BccLattice::addAnchors(unsigned * anchored, Vector3F * pos, unsigned n)
{
	unsigned i=0;
	for(; i< n; i++) addAnchor(anchored, pos[i]);
}

void BccLattice::addAnchor(unsigned * anchored, const Vector3F & pnt)
{
	Vector3F q[4];
	unsigned j, i=0;
	for(; i< numTetrahedrons(); i++) {
        Tetrahedron * tet = &m_tetrahedrons[i];
        for(j=0; j< 4; j++)
            q[j] = nodeCenter(tet->v[j]); 
        
		if(!pointInsideTetrahedronTest(pnt, q)) continue;
		
		for(j=0; j< 4; j++) {
			sdb::CellValue * found = findGrid(tet->v[j]);
			anchored[found->index] = 1;
		}
    }
}

void BccLattice::extractTetrahedronMeshData(Vector3F * points, unsigned * indices)
{
    extractPoints(points);
    extractIndices(indices);
}

void BccLattice::extractPoints(Vector3F * dst)
{
    unsigned i = 0;
    sdb::CellHash * latticeNode = cells();
    latticeNode->begin();
	while(!latticeNode->end()) {
	    if(latticeNode->value()->visited) {
	        dst[i] = nodeCenter(latticeNode->key());
	        i++;
	    }
	    latticeNode->next();
	}
}

void BccLattice::extractIndices(unsigned * dst)
{
    unsigned i, j, k = 0;
	for(i=0; i< numTetrahedrons(); i++) {
        Tetrahedron * tet = &m_tetrahedrons[i];
        for(j=0; j< 4; j++) {
            sdb::CellValue * found = findGrid(tet->v[j]);
            dst[k] = found->index;
            k++;
        }
    }
}
//:~