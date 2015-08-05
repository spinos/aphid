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

LarixInterface::LarixInterface() {}
LarixInterface::~LarixInterface() {}

bool LarixInterface::CreateWorld(LarixWorld * world)
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
	APointCloud * pc = ConvertTetrahedrons(tetra);
	world->setPointCloud(pc);

// a field of color
	const unsigned n = tetra->numPoints();
	AField dv;
	dv.addVec3Channel("foo", n);
	dv.useChannel("foo");
	Vector3F * vdv = dv.vec3Value();
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
	
    AdaptiveField * g = new AdaptiveField(tree.getBBox());
    g->create(&tree, &dv, tetra, 7);
	world->setField(g);
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

void LarixInterface::DrawWorld(LarixWorld * world, KdTreeDrawer * drawer)
{
	// APointCloud * cloud = world->pointCloud();
	//if(!cloud) return;
    
    ATetrahedronMesh * mesh = world->tetrahedronMesh();
    if(!mesh) return;
	
	drawer->setColor(.17f, .21f, .15f);
	//drawer->pointCloud(cloud);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    drawer->tetrahedronMesh(mesh);
	
	drawer->setColor(.3f, .2f, .1f);
	//drawer->cartesianGrid(world->field());
    DrawField(world->field(), "foo", drawer);
}

void LarixInterface::DrawField(AdaptiveField * field, 
                          const std::string & channelName,
                          KdTreeDrawer * drawer)
{
    if(!field->useChannel(channelName)) {
        std::cout<<" field has no channel named "<<channelName;
        return;
    }
    drawer->setWired(0);
    BoundingBox box;
    field->getBounding(box);
    
    Plane clipp;
    clipp.create(box.center(), box.center() + Vector3F::ZAxis,
                 box.center() + Vector3F::XAxis * .5f + Vector3F::ZAxis,
                 box.center() + Vector3F::XAxis + Vector3F::YAxis * 2.f);
    
    Vector3F nor;
    clipp.getNormal(nor);
    Vector3F pop;
    
    Vector3F * col = field->vec3Value();
    sdb::CellHash * c = field->cells();
	Vector3F l;
    float h;
	c->begin();
	while(!c->end()) {
		l = field->cellCenter(c->key());
		h = field->cellSizeAtLevel(c->value()->level);
        
        clipp.projectPoint(l, pop);
        if((l-pop).dot(nor) < 0.f) {
           // if(c->value()->visited ) {

            drawer->setColor(col->x, col->y, col->z);
            drawer->unitCubeAt(l, h);
           // }
        }
		
	    c->next();
        col++;
	}
}
//:~