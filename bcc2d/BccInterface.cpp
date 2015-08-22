/*
 *  BccInterface.cpp
 *  larix
 *
 *  Created by jian zhang on 7/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "BccInterface.h"
#include <GeometryArray.h>
#include <ATriangleMesh.h>
#include <ATetrahedronMeshGroup.h>
#include "BccWorld.h"
#include <KdTreeDrawer.h>
#include "tetrahedron_math.h"
#include <KdIntersection.h>
#include <HesperisFile.h>
#include <BaseLog.h>
#include <boost/timer.hpp>

BccInterface::BccInterface() 
{ 
    m_cells = new BaseBuffer;
}

BccInterface::~BccInterface() 
{ 
    delete m_cells;
}

bool BccInterface::createWorld(BccWorld * world)
{  

	GeometryArray curvegeo;
	if(ReadCurveData(&curvegeo)) {
		const unsigned n = curvegeo.numGeometries();
		std::cout<<"\n bcc interface loading "<<n<<" curve geometries. ";
		unsigned i=0;
		for(;i<n;i++)
			world->addCurveGroup((CurveGroup *)curvegeo.geometry(i));
	}
	else {
		std::cout<<"\n hes file contains no curve geometries. ";
	}
	
    GeometryArray trigeo;
	if(ReadTriangleData(&trigeo)) {
		const unsigned n = trigeo.numGeometries();
		std::cout<<"\n bcc interface loading "<<n<<" triangle mesh geometries. ";
		unsigned i=0;
		for(;i<n;i++)
			world->addTriangleMesh((ATriangleMesh *)trigeo.geometry(i));
	}
	else {
		std::cout<<"\n hes file contains no triangle mesh geometries. ";
	}
	
	world->buildTetrahedronMesh();

	return true;
}

void BccInterface::drawWorld(BccWorld * world, KdTreeDrawer * drawer)
{
	const float das = world->drawAnchorSize();
	const unsigned n = world->numTetrahedronMeshes();
	unsigned i;
	for(i=0;i<n;i++) {
		drawTetrahedronMesh(world->tetrahedronMesh(i), drawer);
		drawAnchors(world->tetrahedronMesh(i), drawer, das);
    
		//drawer->setColor(.17f, .21f, .15f);
		//glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//drawer->tetrahedronMesh(mesh);
	}
	
	drawGeometry(world->triangleGeometries(), drawer);
	
	unsigned igroup;
	GeometryArray * selected = world->selectedGroup(igroup);
	if(selected) {
		drawer->setGroupColorLight(igroup);
		glDisable(GL_DEPTH_TEST);
		drawer->geometry(selected);
	}
		
	// drawer->setColor(.3f, .2f, .1f);
	// drawGrid(world->field(), drawer);
	// drawField(world->field(), "dP", drawer);
	// drawLocateCells(world->field(), mesh, drawer);
}

void BccInterface::drawTetrahedronMesh(ATetrahedronMesh * m, KdTreeDrawer * drawer)
{
	glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor3f(0.51f, 0.53f, 0.52f);
	drawer->tetrahedronMesh(m);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3f(.03f, .14f, .44f);
    drawer->tetrahedronMesh(m);
}

void BccInterface::drawGeometry(GeometryArray * geos, KdTreeDrawer * drawer)
{
	glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor3f(0.63f, 0.64f, 0.65f);
	drawer->geometry(geos);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3f(.03f, .14f, .44f);
    drawer->geometry(geos);
}

void BccInterface::drawAnchors(AGenericMesh * mesh, KdTreeDrawer * drawer,
								float drawSize)
{
	glEnable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	unsigned j;
    glColor3f(.59f, .21f, 0.f);
	Vector3F * p = (Vector3F *)mesh->points();
	unsigned * a = (unsigned *)mesh->anchors();
	const unsigned n = mesh->numPoints();
	for(j=0;j<n;j++) {
		if(a[j]>0) drawer->alignedDisc(p[j], drawSize);
	}
}

bool BccInterface::saveWorld(BccWorld * world)
{
	HesperisFile hes;
	hes.setWriteComponent(HesperisFile::WTetra);
	
	ATetrahedronMeshGroup * cm = world->combinedTetrahedronMesh();
	hes.addTetrahedron("tetra_c", cm);

	if(!hes.open(FileName)) return false;
	hes.setDirty();
	hes.save();
	hes.close();

	delete cm;
    
	return true;
}
//:~