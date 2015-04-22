#include "BccGrid.h"
#include <BezierCurve.h>
#include <GeoDrawer.h>
#include <BccLattice.h>
#include "bcc_common.h"

BccGrid::BccGrid(const BoundingBox & bound) :
    CartesianGrid(bound)
{
    m_lattice = new BccLattice(bound);
}

BccGrid::~BccGrid() 
{
    delete m_lattice;
}

void BccGrid::create(BezierSpline * splines, unsigned n, int maxLevel)
{
	m_splines = splines;
	m_numSplines = n;
	
// start at 8 cells per axis
    int level = 3;
    const int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;
    m_tolerance = 0.1f;
    const Vector3F ori = origin() + Vector3F(hh, hh, hh) * .999f;
    Vector3F sample, closestP;
    BoundingBox box;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* (float)i, h* (float)j, h* (float)k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
                if(intersectBox(box))
                    addCell(sample, level);
            }
        }
    }
    std::cout<<" n level 3 cell "<<numCells()<<"\n";
	
    if(maxLevel <=5 ) {
		maxLevel = 5;
		std::cout<<" max level cannot < 5\n";
	}
	if(maxLevel >= 9) {
		maxLevel = 9;
		std::cout<<" max level cannot > 9\n";
	}
	
    for(level=4; level<= maxLevel; level++)
        subdivide(level);
    
	// printHash();
	std::cout<<" creating bcc lattice\n";
	createLatticeNode();
	std::cout<<" creating bcc tetrahedrons\n";
	createLatticeTetrahedron();
	m_lattice->countVisitedNodes();
	
	m_lattice->logTetrahedronMesh();
	std::cout<<" n green edges "<<m_lattice->numGreenEdges()<<"\n";
	std::cout<<" n tetrahedrons "<<m_lattice->numTetrahedrons()<<"\n";
	std::cout<<" n vertices "<<m_lattice->numVertices()<<"\n";
}

void BccGrid::subdivide(int level)
{
	sdb::CellHash * c = cells();
	
    const unsigned n = c->size();
	
	unsigned * parentKey = new unsigned[n];
	unsigned i = 0;
	c->begin();
	while(!c->end()) {
		parentKey[i] = c->key();
		i++;
		c->next();
	}
	
    int u;
    Vector3F sample, subs, closestP;
    BoundingBox box;
    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;
    for(i=0; i< n; i++) {
        sample = cellCenter(parentKey[i]);
		removeCell(parentKey[i]);
		for(u = 0; u < 8; u++) {
			subs = sample + Vector3F(hh * OctChildOffset[u][0], 
			hh * OctChildOffset[u][1], 
			hh * OctChildOffset[u][2]);

			box.setMin(subs.x - hh, subs.y - hh, subs.z - hh);
			box.setMax(subs.x + hh, subs.y + hh, subs.z + hh);

			if(intersectBox(box))
			   addCell(subs, level);
		}
    }
    delete[] parentKey;
	std::cout<<" n level "<<level<<" cell "<<numCells()<<"\n";
}

void BccGrid::createLatticeNode()
{
    Vector3F cen;
    float h;
    sdb::CellHash * c = cells();
	c->begin();
	while(!c->end()) {
		cen = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level);
		m_lattice->add38Node(cen, h);
	    c->next();
	}
	m_lattice->prepareTetrahedron();
}

void BccGrid::createLatticeTetrahedron()
{
    Vector3F cen;
    float h;
    sdb::CellHash * c = cells();
	c->begin();
	while(!c->end()) {
		cen = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level);
		m_lattice->touchIntersectedTetrahedron(cen, h, m_splines, m_numSplines);
	    c->next();   
	}
	m_lattice->untouchGreenEdges();
	c->begin();
	while(!c->end()) {
		cen = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level);
		m_lattice->add24Tetrahedron(cen, h);
	    c->next();   
	}
	c->begin();
	while(!c->end()) {
		cen = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level);
		m_lattice->addNeighborTetrahedron(cen, h);
	    c->next();   
	}
}

void BccGrid::draw(GeoDrawer * drawer, unsigned * anchored)
{
	sdb::CellHash * c = cells();
	Vector3F l;
    BoundingBox box;
    float h;
    
    drawer->setColor(0.3f, .2f, 0.1f);
    
	c->begin();
	while(!c->end()) {
		l = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level) * .5f;
        box.setMin(l.x - h, l.y - h, l.z - h);
        box.setMax(l.x + h, l.y + h, l.z + h);
        drawer->boundingBox(box);
		
	    c->next();   
	}

	m_lattice->draw(drawer, anchored);
}

void BccGrid::drawHash()
{
	sdb::CellHash * c = cells();
	Vector3F l;
	glColor3f(0.f, .1f, .4f);
    
	glBegin(GL_LINE_STRIP);
	c->begin();
	while(!c->end()) {
		l = cellCenter(c->key());
		glVertex3fv((GLfloat *)&l);
	    c->next();   
	}
	glEnd();
}

bool BccGrid::intersectBox(const BoundingBox & box) const
{
	unsigned i = 0;
	for(; i<m_numSplines; i++) {
		if(BezierCurve::intersectBox(m_splines[i], box))
			return true;
	}
	return false;
}

void BccGrid::addAnchors(unsigned * anchors, Vector3F * pos, unsigned n)
{
	m_lattice->addAnchors(anchors, pos, n);
}

const unsigned BccGrid::numTetrahedronVertices() const
{
	return m_lattice->numVertices();
}
//:!