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
#include <MeshSeparator.h>
#include <PrincipalComponents.h>
#include "BlockBccMeshBuilder.h"
#include <BaseLog.h>
#include <boost/timer.hpp>

BccInterface::BccInterface() 
{
	m_patchSeparator = new MeshSeparator;
	m_patchMesh = NULL;
    m_tetMesh = NULL;
    // testBlockMesh();
}

BccInterface::~BccInterface() 
{
	delete m_patchSeparator;
}

bool BccInterface::createWorld(BccWorld * world)
{
	bool hasCrv = loadCurveGeometry(world, FileName);
	bool hasTri = loadTriangleGeometry(world, FileName);
	
	if(hasCrv && hasTri) 
		world->buildTetrahedronMesh();

	return true;
}

bool BccInterface::loadTriangleGeometry(BccWorld * world, const std::string & filename)
{
	const std::string oldFilename = FileName;
	FileName = filename;
	GeometryArray * trigeo = new GeometryArray;
	bool res = ReadTriangleData(trigeo);
	if(res) {
		const unsigned n = trigeo->numGeometries();
		std::cout<<"\n bcc interface loading "<<n<<" triangle mesh geometries. ";
		world->setTiangleGeometry(trigeo);
	}
	else {
		std::cout<<"\n hes file contains no triangle mesh geometry. ";
	}
	FileName = oldFilename;
	return res;
}

bool BccInterface::loadCurveGeometry(BccWorld * world, const std::string & filename)
{
	const std::string oldFilename = FileName;
	FileName = filename;
	GeometryArray curvegeo;
	bool res = ReadCurveData(&curvegeo);
	if(res) {
		const unsigned n = curvegeo.numGeometries();
		std::cout<<"\n bcc interface loading "<<n<<" curve geometries. ";
		unsigned i=0;
		for(;i<n;i++)
			world->addCurveGroup((CurveGroup *)curvegeo.geometry(i));
	}
	else {
		std::cout<<"\n hes file contains no curve geometry. ";
	}
	FileName = oldFilename;
	return res;
}

bool BccInterface::loadPatchGeometry(BccWorld * world, const std::string & filename)
{
	const std::string oldFilename = FileName;
	FileName = filename;
	GeometryArray * trigeo = new GeometryArray;
	bool res = ReadTriangleData(trigeo);
	if(res) {
		const unsigned n = trigeo->numGeometries();
		std::cout<<"\n bcc interface loading "<<n<<" triangle mesh geometries as patch ";
		m_patchMesh = (ATriangleMesh *)trigeo->geometry(0);
		unsigned i=0;
		for(;i<n;i++) {
            std::vector<AOrientedBox> patchBoxes;
			if(separate((ATriangleMesh *)trigeo->geometry(i),
                patchBoxes)) {
				world->addPatchBoxes(patchBoxes);
                patchBoxes.clear();
            }
		}
	}
	else {
		std::cout<<"\n hes file contains no triangle mesh geometry. ";
	}
	FileName = oldFilename;
	return res;
}

void BccInterface::drawWorld(BccWorld * world, KdTreeDrawer * drawer)
{
	const float das = world->drawAnchorSize();
	const unsigned n = world->numTetrahedronMeshes();
	unsigned i;
	for(i=0;i<n;i++) {
		drawTetrahedronMesh(world->tetrahedronMesh(i), drawer);
		drawAnchors(world->tetrahedronMesh(i), drawer, das);
	}
	
	if(world->triangleGeometries()) drawGeometry(world->triangleGeometries(), drawer);
	
	unsigned igroup;
	GeometryArray * selected = world->selectedGroup(igroup);
	if(selected) {
		drawer->setGroupColorLight(igroup);
		glDisable(GL_DEPTH_TEST);
		drawer->geometry(selected);
	}
	
	if(m_patchMesh) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glColor3f(.03f, .14f, .44f);
		drawer->geometry(m_patchMesh);
	}
    
    if(m_tetMesh) {
#if 0
		drawTetrahedronMesh(m_tetMesh, drawer);
#else
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glColor3f(.03f, .14f, .44f);
		drawer->tetrahedronMesh(m_tetMesh);
#endif
	}
	
    const std::vector<AOrientedBox> * boxes = world->patchBoxes();
	glColor3f(0.f, 0.99f, 0.f);
	glBegin(GL_LINES);
	std::vector<AOrientedBox>::const_iterator it = boxes->begin();
	for(;it!=boxes->end();++it)
		drawer->orientedBox(&(*it));
	glEnd();
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
    if(world->numTetrahedronMeshes() < 1) {
        std::cout<<" no tetrahedron mesh to save";
        return false;
    }
    
	HesperisFile hes;
    if(!hes.open(FileName)) {
        
        GeometryArray * gmgeo = world->triangleGeometries();
        if(!gmgeo) {
            std::cout<<" no triangle mesh to save";
            return false;
        }
        
        hes.create(FileName);
        hes.setWriteComponent(HesperisFile::WTri);
        hes.addTriangleMesh("growmesh", (ATriangleMeshGroup *)gmgeo->geometry(0));
        hes.setDirty();
        hes.save();
	}
    
	hes.setWriteComponent(HesperisFile::WTetra);
	
	unsigned ntet, nvert, nstripe, nanchor;
	world->computeTetrahedronMeshStatistics(ntet, nvert, nstripe, nanchor);
	
	ATetrahedronMeshGroup * cm = new ATetrahedronMeshGroup;
	cm->create(nvert, ntet, nstripe);
	
	const unsigned n = world->numTetrahedronMeshes();
	
	ntet = 0;
	nvert = 0;
    nstripe = 0;
	unsigned i = 0;
	for(; i < n; i++) {
		ATetrahedronMeshGroup * imesh = world->tetrahedronMesh(i);
        cm->copyPointDrift(imesh->pointDrifts(), 
                              imesh->numStripes(), 
                              nstripe,
                              nvert);
        cm->copyIndexDrift(imesh->indexDrifts(), 
                              imesh->numStripes(), 
                              nstripe,
                              ntet*4);
        
		cm->copyStripe(imesh, nvert, ntet * 4);
		
		ntet += imesh->numTetrahedrons();
		nvert += imesh->numPoints();
        nstripe += imesh->numStripes();
	}
	float vlm = cm->calculateVolume();
	cm->setVolume(vlm);
	
	std::cout<<"\n combined all for save out ";
	cm->verbose();
	
	hes.addTetrahedron("tetra_c", cm);
	hes.setDirty();
	hes.save();
	hes.close();

	delete cm;
    
	return true;
}

bool BccInterface::separate(ATriangleMesh * mesh, std::vector<AOrientedBox> & patchBoxes)
{
	std::cout<<"\n mesh n tri "<<mesh->numTriangles();
	m_patchSeparator->separate(mesh);
	const unsigned n = m_patchSeparator->numPatches();
	if(n < 2) {
		std::cout<<"\n cannot separate one continuous mesh ";
		return false;
	}
	
	std::cout<<"\n separate to "<<n<<" patches ";
	
	PrincipalComponents pca;
	BaseBuffer pos;
	m_patchSeparator->patchBegin();
	while(!m_patchSeparator->patchEnd()) {
		const unsigned np = m_patchSeparator->getPatchCvs(&pos, mesh);
		m_patchSeparator->nextPatch();
		
		patchBoxes.push_back( pca.analyze((Vector3F *)pos.data(), np) );
	}
	
	return true;
}

void BccInterface::testBlockMesh()
{
    BlockBccMeshBuilder builder;
    AOrientedBox box1;
    box1.setExtent(Vector3F(28.f, 14.f, 1.f));
    box1.setCenter(Vector3F(0.1f, -2.f, 1.f));
	
	AOrientedBox box2;
	Matrix33F rot; rot.set(Quaternion(0.f, 1.f, 0.f, .5f));
	box2.setOrientation(rot);
    box2.setExtent(Vector3F(18.f, 4.f, 1.f));
    box2.setCenter(Vector3F(2.1f, 13.f, -1.f));
	
	GeometryArray boxes;
	boxes.create(2);
	boxes.setGeometry(&box1, 0);
	boxes.setGeometry(&box2, 1);
	unsigned nv, nt, ns;
    builder.build(&boxes, nt, nv, ns);
	std::cout<<"\n tet nt nv ns "<<nt<<" "<<nv<<" "<<ns;
	m_tetMesh = new ATetrahedronMeshGroup;
	m_tetMesh->create(nv, nt, ns);
	builder.getResult(m_tetMesh);
}
//:~