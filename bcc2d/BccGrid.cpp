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

void BccGrid::create(BezierCurve * curve, int maxLevel)
{
// start at 8 cells per axis
    int level = 3;
    const int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;
    m_tolerance = 0.1f;
    const Vector3F ori = origin() + Vector3F(hh, hh, hh);
    Vector3F sample, closestP;
    BoundingBox box;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* i, h* j, h* k);
                box.setMin(sample.x - hh, sample.y - hh, sample.z - hh);
                box.setMax(sample.x + hh, sample.y + hh, sample.z + hh);
                if(curve->intersectBox(box))
                    addCell(sample, level);
            }
        }
    }
    std::cout<<" n level 3 cell "<<numCells()<<"\n";
    for(level=4; level<= maxLevel; level++)
        subdivide(curve, level);
	// printHash();
	createLatticeNode();
}

void BccGrid::subdivide(BezierCurve * curve, int level)
{
	sdb::MortonHash * c = cells();
	
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

			if(curve->intersectBox(box))
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
    sdb::MortonHash * c = cells();
	c->begin();
	while(!c->end()) {
		cen = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level);
		m_lattice->addOctahedron(cen, h);
	    c->next();   
	}	
}

void BccGrid::draw(GeoDrawer * drawer)
{
	sdb::MortonHash * c = cells();
	Vector3F l;
    BoundingBox box;
    float h;
    
    drawer->setColor(0.f, .3f, 0.2f);
    
	c->begin();
	while(!c->end()) {
		l = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level) * .5f;
        box.setMin(l.x - h, l.y - h, l.z - h);
        box.setMax(l.x + h, l.y + h, l.z + h);
        drawer->boundingBox(box);
		
	    c->next();   
	}

	m_lattice->draw(drawer);
}

void BccGrid::drawHash()
{
	sdb::MortonHash * c = cells();
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

