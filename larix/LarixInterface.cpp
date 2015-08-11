/*
 *  LarixInterface.cpp
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "LarixInterface.h"
#include <GeometryArray.h>
#include <ATetrahedronMesh.h>
#include <APointCloud.h>
#include "LarixWorld.h"
#include <KdTreeDrawer.h>
#include "tetrahedron_math.h"
#include <KdIntersection.h>
#include "AdaptiveField.h"
#include <Plane.h>
#include <BaseLog.h>

#if 0
BaseLog lxlg("log.txt");
#endif

LarixInterface::LarixInterface() 
{ 
    m_cells = new BaseBuffer;
}

LarixInterface::~LarixInterface() 
{ 
    delete m_cells;
}

bool LarixInterface::createWorld(LarixWorld * world)
{  
	GeometryArray geos;
	if(!ReadTetrahedronData(&geos)) return false;
 
	ATetrahedronMesh * tetra = (ATetrahedronMesh *)geos.geometry(0);
    
    KdIntersection tree;
    tree.addGeometry(tetra);
    KdTree::MaxBuildLevel = 20;
	KdTree::NumPrimitivesInLeafThreashold = 9;
    tree.create();
    
	world->setTetrahedronMesh(tetra);

    AdaptiveField * g = new AdaptiveField(tree.getBBox());
    g->create(&tree, tetra, 7);
    g->addVec3Channel("dP");

    if(world->hasSourceP()) {
        TypedBuffer * sourceP = world->sourceP();
        *sourceP -= tetra->pointsBuf();
        g->computeChannelValue("dP", sourceP, tetra->sampler());
        
        //lxlg.writeVec3(sourceP, sourceP->numElements(), "P");
    }
    else {
        const unsigned n = tetra->numPoints();
        TypedBuffer * dv = new TypedBuffer;
        dv->create(TypedBuffer::TVec3, n*12);
    
        Vector3F * vdv = dv->typedData<Vector3F>();
        Vector3F * p = tetra->points();
        const BoundingBox bigBox = tree.getBBox();
        const float boxdx = bigBox.distance(0);
        const float boxdy = bigBox.distance(1);
        const float boxdz = bigBox.distance(2);
        unsigned i = 0;
        for(;i<n;i++) {
            vdv[i].x = (p[i].x - bigBox.getMin().x) / boxdx;
            vdv[i].y = (p[i].y - bigBox.getMin().y) / boxdy;
            vdv[i].z = (p[i].z - bigBox.getMin().z) / boxdz;
        }
        world->setSourceP(dv);
        g->computeChannelValue("dP", dv, tetra->sampler());

        //lxlg.writeVec3(g->namedChannel("dP"), g->namedChannel("dP")->numElements(), "dP");
    }
	world->setField(g);
    buildCells(g);
    
    world->beginCache();
    
    // testLocateCells(g, tetra);
#if 0
	BaseBuffer grids;
	g->printGrids(&grids);
	lxlg.writeInt3(&grids, g->numCells(), "grd");
#endif	
	return true;
}

APointCloud * LarixInterface::ConvertTetrahedrons(ATetrahedronMesh * mesh)
{
	mesh->verbose();
	const unsigned nv = mesh->numPoints();
	APointCloud * pc = new APointCloud;
	pc->create(nv);
	pc->copyPointsFrom(mesh->points());
	
	float * r = pc->pointRadius();
	unsigned i = 0;
	for(;i<nv;i++) r[i] = 0.f;
	
	Vector3F q[4];
	float tvol;
	Vector3F * p = mesh->points();
	const unsigned nt = mesh->numTetrahedrons();
	for(i=0;i<nt;i++) {
		unsigned * v = mesh->tetrahedronIndices(i);
		q[0] = p[v[0]];
		q[1] = p[v[1]];
		q[2] = p[v[2]];
		q[3] = p[v[3]];
		tvol = tetrahedronVolume(q);
		tvol *= .25f;
        r[v[0]] += tvol;
		r[v[1]] += tvol;
		r[v[2]] += tvol;
		r[v[3]] += tvol;
	}
// VolumeOfSphere = 4 Pi r^3 / 3
// RadiusOfSphere = ( 3 V / 4 / Pi )^(1/3)	
	for(i=0;i<nv;i++) r[i] = pow(r[i] * .75f / 3.14159f, .33f);
	return pc;
}

void LarixInterface::drawWorld(LarixWorld * world, KdTreeDrawer * drawer)
{
    ATetrahedronMesh * mesh = world->tetrahedronMesh();
    if(!mesh) return;
	
	drawer->setColor(.17f, .21f, .15f);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    drawer->tetrahedronMesh(mesh);
	
	drawer->setColor(.3f, .2f, .1f);
	// drawGrid(world->field(), drawer);
	drawField(world->field(), "dP", drawer);
	//drawLocateCells(world->field(), mesh, drawer);
}

void LarixInterface::drawLocateCells(AdaptiveField * fld, ATetrahedronMesh * mesh,
					KdTreeDrawer * drawer)
{
	drawer->setColor(1.f, 0.f, 0.f);
	glBegin(GL_TRIANGLES);

	const unsigned n = mesh->numPoints();
    Vector3F * p = mesh->points();
    unsigned i = 0;
	for(;i<n;i++) {
        sdb::CellValue * c = fld->locateCell(p[i]);
        if(!c) {
            std::cout<<"\n cannot find cell for p["<< i << "] " <<p[i];
            //lxlg.writeMortonCode(fld->mortonEncodeLevel(p[i], 7));
            drawer->unitCubeAt(p[i], .5f);
        }
	}
	glEnd();    
}

void LarixInterface::drawField(AdaptiveField * field, 
                          const std::string & channelName,
                          KdTreeDrawer * drawer)
{
    drawer->setWired(0);
    BoundingBox box;
    field->getBounding(box);
    Plane clipp;
    clipp.set(Vector3F::XAxis + Vector3F::YAxis + Vector3F::ZAxis,
              box.center());
    
    Vector3F nor;
    clipp.getNormal(nor);
    Vector3F pop;

    const unsigned n = field->numCells();
    Vector3F * col = (Vector3F *)field->namedData("dP");
    unsigned i;
    
    Float4 * src = (Float4 *)m_cells->data();
    Vector3F l;
    float h;

    glBegin(GL_TRIANGLES);
    for(i=0;i<n;i++) {
        l.set(src[i].x, src[i].y, src[i].z);
        h = src[i].w;
        clipp.projectPoint(l, pop);
         if((l-pop).dot(nor) < 0.f) { // behind clipping plane
            glColor3f(col->x, col->y, col->z);
            drawer->unitCubeAt(l, h);
            
        }
        col++;
    }
    glEnd();
}

void LarixInterface::drawGrid(AdaptiveField * field,
								KdTreeDrawer * drawer)
{
	const unsigned n = field->numCells();
    unsigned i = 0;
	
    Float4 * src = (Float4 *)m_cells->data();
    Vector3F l;
    float h;
    
	glBegin(GL_LINES);
    for(i=0;i<n;i++) {
        l.set(src[i].x, src[i].y, src[i].z);
        h = src[i].w;
        
		drawer->unitBoxAt(l, h);
    }
	glEnd();
}

void LarixInterface::buildCells(AdaptiveField * field)
{
    const unsigned n = field->numCells();
    m_cells->create(n*16);
    
    BaseBuffer codes;
    codes.create(n*4);
    unsigned * dcode = (unsigned *)codes.data();
    
    Float4 * dst = (Float4 *)m_cells->data();
    
    sdb::CellHash * c = field->cells();
	Vector3F l;
    float h;
    unsigned i = 0;
	c->begin();
	while(!c->end()) {
		l = field->cellCenter(c->key());
        dcode[i] = c->key();
		h = field->cellSizeAtLevel(c->value()->level);
        
        dst[i].x = l.x;
        dst[i].y = l.y;
        dst[i].z = l.z;
        dst[i].w = h;

	    c->next();
        i++;
	}
#if 0
    lxlg.writeMortonCode(&codes, n, "key");
#endif
}

void LarixInterface::testLocateCells(AdaptiveField * fld, ATetrahedronMesh * tetra)
{
    std::cout<<"\n test locate cells ... ";
    const unsigned n = tetra->numPoints();
    Vector3F * p = tetra->points();
	Vector3F q;
	bool passed = true;
    unsigned i = 0;
	for(;i<n;i++) {
		q = p[i];
		q += Vector3F(RandomFn11(), RandomFn11(), RandomFn11());
		fld->putPInsideBound(q);
        sdb::CellValue * c = fld->locateCell(q);
        if(!c) {
            std::cout<<"\n cannot find cell for p["<< i << "] " <<q;
            //lxlg.writeMortonCode(fld->mortonEncodeLevel(p[i], 7));
			passed = false;
        }
	}
	if(passed) std::cout<<"all passed.\n";
}
//:~