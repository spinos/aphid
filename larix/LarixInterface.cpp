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

BaseLog lxlg("log.txt");

LarixInterface::LarixInterface() 
{ 
    m_cells = new BaseBuffer;
    m_cellColors = new BaseBuffer;
}

LarixInterface::~LarixInterface() 
{ 
    delete m_cells;
    delete m_cellColors;
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
    g->create(&tree, tetra);
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
    world->setCellColorOutBuf(m_cellColors);
    m_cellColors->copyFrom(g->namedChannel("dP")->data());
    //lxlg.writeVec3(m_cellColors, g->namedChannel("dP")->numElements(), "dP");
    world->beginCache();
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
	//drawer->cartesianGrid(world->field());
    
    //m_cellColors->copyFrom(world->field()->namedChannel("dP")->data());
    drawField(world->field(), "dP", drawer);
}

void LarixInterface::drawField(AdaptiveField * field, 
                          const std::string & channelName,
                          KdTreeDrawer * drawer)
{
    //if(!field->useChannel(channelName)) {
   //     std::cout<<" field has no channel named "<<channelName;
   //     return;
   // }
    drawer->setWired(0);
    BoundingBox box;
    field->getBounding(box);
    
    Plane clipp;
    clipp.set(Vector3F::XAxis + Vector3F::YAxis + Vector3F::ZAxis,
              box.center() - Vector3F::XAxis * .5f);
    
    Vector3F nor;
    clipp.getNormal(nor);
    Vector3F pop;
    
    
    const unsigned n = field->numCells();
    std::cout<<"begin draw col"<<n;
    m_cellColors->copyFrom(field->namedChannel("dP")->data());
    Vector3F * col = (Vector3F *)m_cellColors->data(); //field->vec3Value();
    
    //lxlg.writeVec3(m_cellColors, n, "dP", BaseLog::FAlways);
    //Vector3F * srcf = (Vector3F *)field->namedChannel("dP")->data();
    unsigned i = 0;
    //for(;i<n;i++) col[i] = srcf[i];
    
    Float4 * src = (Float4 *)m_cells->data();
    Vector3F l;
    float h;
    
    for(i=0;i<n;i++) {
        l.set(src[i].x, src[i].y, src[i].z);
        h = src[i].w;
        clipp.projectPoint(l, pop);
        if((l-pop).dot(nor) < 0.f) { // behind clipping plane
            //drawer->setColor(col[i].x, col[i].y, col[i].z);
            glColor3f(col->x, col->y, col->z);
            if((i&1023) == 0)std::cout<<" "<<l;
            
            drawer->unitCubeAt(l, h);
            
        }
        if((i&1023) == 0) std::cout<<" "<<i;
        col++;
    }

    std::cout<<"end draw col";
}

void LarixInterface::buildCells(AdaptiveField * field)
{
    const unsigned n = field->numCells();
    std::cout<<"\n buffer n cells "<<n;
    m_cells->create(n*16);
    m_cellColors->create(n*12);
    
    Float4 * dst = (Float4 *)m_cells->data();
    
    sdb::CellHash * c = field->cells();
	Vector3F l;
    float h;
    unsigned i = 0;
	c->begin();
	while(!c->end()) {
		l = field->cellCenter(c->key());
		h = field->cellSizeAtLevel(c->value()->level);
        
        dst[i].x = l.x;
        dst[i].y = l.y;
        dst[i].z = l.z;
        dst[i].w = h;

	    c->next();
        i++;
	}
}
//:~