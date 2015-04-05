#include "BccGrid.h"
#include <BezierCurve.h>
#include <GeoDrawer.h>
BccGrid::BccGrid(const BoundingBox & bound) :
    CartesianGrid(bound, 8)
{

}

BccGrid::~BccGrid() {}

void BccGrid::create(BezierCurve * curve)
{
    int level = 3;
    int dim = 1<<level;
    int i, j, k;

    const float h = cellSizeAtLevel(level);
    m_tolerance = 0.33f;
    const Vector3F ori = origin() + Vector3F(h*.5f, h*.5f, h*.5f);
    Vector3F sample, closestP;
    BoundingBox box;
    for(k=0; k < dim; k++) {
        for(j=0; j < dim; j++) {
            for(i=0; i < dim; i++) {
                sample = ori + Vector3F(h* i, h* j, h* k);
                box.setMin(sample.x - h*.5f, sample.y - h*.5f, sample.z - h*.5f);
                box.setMax(sample.x + h*.5f, sample.y + h*.5f, sample.z + h*.5f);
                if(curve->intersectBox(box))
                    addCell(sample, level);
            }
        }
    }
    std::cout<<" n level 3 cell "<<numCells()<<"\n";
	subdivide(curve, 4);
	
	// subdivide(curve, 5);
	// subdivide(curve, 6);
	// subdivide(curve, 7);
	// subdivide(curve, 8);
	// printHash();
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
	
    int u, v, w;
    Vector3F sample, subs, closestP;
    BoundingBox box;
    const float h = cellSizeAtLevel(level);
    const float hh = h * .5f;
    int isFirst;
    for(i=0; i< n; i++) {
        sample = cellCenter(parentKey[i]);
		removeCell(parentKey[i]);
		for(u = -1; u <= 1; u+=2) {
           for(v = -1; v <= 1; v+=2) {
               for(w = -1; w <= 1; w+=2) {
                   subs = sample + Vector3F(hh * u, hh * v, hh * w);
                   
				   box.setMin(subs.x - hh, subs.y - hh, subs.z - hh);
                   box.setMax(subs.x + hh, subs.y + hh, subs.z + hh);
                
                  if(curve->intersectBox(box))
                       addCell(subs, level);
               }
           } 
        }
		
    }
    delete[] parentKey;
	std::cout<<" n level "<<level<<" cell "<<numCells()<<"\n";
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
		unsigned k = c->key();
		l = cellCenter(c->key());
		h = cellSizeAtLevel(c->value()->level) * .48f;
        box.setMin(l.x - h, l.y - h, l.z - h);
        box.setMax(l.x + h, l.y + h, l.z + h);
        drawer->boundingBox(box);
		
	    c->next();   
	}	
}

void BccGrid::drawHash()
{
	sdb::MortonHash * c = cells();
	Vector3F l;
	glColor3f(0.f, .1f, .4f);
    
	glBegin(GL_LINE_STRIP);
	c->begin();
	while(!c->end()) {
		unsigned k = c->key();
		l = cellCenter(c->key());
		glVertex3fv((GLfloat *)&l);
	    c->next();   
	}
	glEnd();
}
