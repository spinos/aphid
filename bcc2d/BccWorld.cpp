#include "BccWorld.h"
#include <BezierCurve.h>
#include <KdTreeDrawer.h>
#include "BccGrid.h"
#include <CurveGroup.h>
#include "bcc_common.h"
#include <line_math.h>
#include <HesperisFile.h>
#include <KdCluster.h>
#include <KdIntersection.h>
#include <GeometryArray.h>
#include <CurveBuilder.h>
#include <BezierCurve.h>
#include <RandomCurve.h>
#include <bezierPatch.h>
#include <APointCloud.h>
#include <BccMesh.h>
#include <FitBccMesh.h>
#include <ATriangleMesh.h>
#include "FitBccMeshBuilder.h"
#include "CurveReduction.h"

BccWorld::BccWorld(KdTreeDrawer * drawer)
{
    m_numCurves = 0;
    m_totalCurveLength = 0.f;
    m_estimatedNumGroups = 2500.f;
	m_drawer = drawer;
	m_curves = new CurveGroup;
    m_reducer = new CurveReduction;
	if(!createCurveGeometryFromFile())
#if WORLD_TEST_SINGLE
		createTestCurveGeometry();
#else
		createRandomCurveGeometry();
#endif
	
	createCurveStartP();
	createAnchorIntersect();
	
	std::cout<<" creating kd tree\n";
	m_cluster = new KdCluster;
	m_cluster->addGeometry(m_allGeo);
	
	KdTree::MaxBuildLevel = 6;
	KdTree::NumPrimitivesInLeafThreashold = 31;
	
	m_cluster->create();
	createTriangleMeshesFromFile();
	createTetrahedronMeshes();
	std::cout<<" done initialize.\n";
}

BccWorld::~BccWorld() 
{
    delete m_reducer;
	delete m_curves;
	delete m_cluster;
	delete m_anchorIntersect;
	delete m_allGeo;
	delete m_curveStartP;
	delete[] m_meshes;
}

bool BccWorld::createCurveGeometryFromFile()
{
	if(!readCurveDataFromFile()) return false;
	
	m_allGeo = new GeometryArray;
	m_allGeo->create(m_curves->numCurves());
	
	const unsigned n = m_curves->numCurves();
	m_allGeo = new GeometryArray;
	m_allGeo->create(n);
	
	unsigned * cc = m_curves->counts();
	Vector3F * cvs = m_curves->points();
	
	m_totalCurveLength = 0.f;
	CurveBuilder cb;
	
	unsigned ncv;
	unsigned cvDrift = 0;
	
	unsigned i, j;
	for(i=0; i< n; i++) {
		
		ncv = cc[i];
		
		for(j=0; j < ncv; j++)			
			cb.addVertex(cvs[j + cvDrift]);
		
		BezierCurve * c = new BezierCurve;
		cb.finishBuild(c);
		
		m_totalCurveLength += c->length();
		
		m_allGeo->setGeometry(c, i);
		
		cvDrift += ncv;
	}
    std::cout<<" n curves "<<n
	    <<" total curve length: "<<m_totalCurveLength<<"\n";
    m_numCurves = n;
	return true;
}

void BccWorld::createTestCurveGeometry()
{
	std::cout<<" gen 1 test curve";
	m_curves->create(1, 9);
	m_curves->counts()[0] = 9;
	
	Vector3F * cvs = m_curves->points();
	cvs[0].set(8.f + RandomFn11(), 1.f + RandomFn11(), 4.1f);
    cvs[1].set(2.f + RandomFn11(), 9.4f + RandomFn11(), 1.11f);
    cvs[2].set(14.f + RandomFn11(), 8.4f + RandomFn11(), -3.13f);
    cvs[3].set(12.f + RandomFn11(), 1.4f + RandomFn11(), 1.14f);
    cvs[4].set(19.f + RandomFn11(), 2.4f + RandomFn11(), 2.16f);
    cvs[5].set(20.f + RandomFn11(), 3.4f + RandomFn11(), 5.17f);
    cvs[6].set(18.f + RandomFn11(), 12.2f + RandomFn11(), 3.18f);
    cvs[7].set(12.f + RandomFn11(), 12.2f + RandomFn11(), 2.19f);
    cvs[8].set(13.f + RandomFn11(), 8.2f + RandomFn11(), -2.18f);
    
    for(unsigned i=0; i<9;i++) {
        cvs[i] -= Vector3F(12.f, 0.f, 0.f);
        cvs[i] *= 3.f;
    }
	
	m_allGeo = new GeometryArray;
	m_allGeo->create(1);
	
	CurveBuilder cb;
	
	unsigned i;
	for(i=0; i< 9; i++)
		cb.addVertex(cvs[i]);
	
	BezierCurve * c = new BezierCurve;
	cb.finishBuild(c);
	m_allGeo->setGeometry(c, 0);
}

void BccWorld::createRandomCurveGeometry()
{
	const unsigned n = 15 * 15;
	m_allGeo = new GeometryArray;
	m_allGeo->create(n);
	
	BezierPatch bp;
	bp.resetCvs();
	
	int i=0;
	bp._contorlPoints[0].y += -.2f;
	bp._contorlPoints[1].y += -.4f;
	bp._contorlPoints[2].y += -.4f;
	bp._contorlPoints[3].y += -.5f;
	
	bp._contorlPoints[4].y += -.5f;
	bp._contorlPoints[5].y += .1f;
	bp._contorlPoints[6].y += .5f;
	bp._contorlPoints[7].y += .1f;
	
	bp._contorlPoints[9].y += .5f;
	bp._contorlPoints[10].y += .5f;
	
	bp._contorlPoints[13].y += -.4f;
	bp._contorlPoints[14].y += -.85f;
	bp._contorlPoints[15].y += -.21f;
	
	i=0;
	for(;i<16;i++) {
		bp._contorlPoints[i] *= 80.f;
		bp._contorlPoints[i].y += 10.f;
		bp._contorlPoints[i].z -= 10.f;
	}
	
	RandomCurve rc;
	rc.create(m_allGeo, 15, 15,
				&bp,
				Vector3F(-.15f, 1.f, 0.33f), 
				11, 21,
				.9f);
}

void BccWorld::createCurveStartP()
{
	const unsigned n = m_allGeo->numGeometries();
	m_curveStartP = new APointCloud;
	m_curveStartP->create(n);
	Vector3F * p = m_curveStartP->points();
	
	unsigned i=0;
	for(;i<n;i++) p[i] = ((BezierCurve *)m_allGeo->geometry(i))->m_cvs[0];	
}

void BccWorld::createAnchorIntersect()
{
	m_anchorIntersect = new KdIntersection;
	m_anchorIntersect->addGeometry(m_curveStartP);
	KdTree::MaxBuildLevel = 32;
	KdTree::NumPrimitivesInLeafThreashold = 9;
	m_anchorIntersect->create();
}

void BccWorld::createTetrahedronMeshes()
{
    if(totalCurveLength()<1.f) {
        std::cout<<" invalid total curve length "<<totalCurveLength()<<" !\n";
        return;
    }
#if WORLD_USE_FIT
    FitBccMeshBuilder::EstimatedGroupSize = totalCurveLength() / m_estimatedNumGroups;
    std::cout<<"\n estimate group size "<<FitBccMeshBuilder::EstimatedGroupSize;
#endif	
	unsigned n = m_cluster->numGroups();
#if WORLD_USE_FIT
	m_meshes = new FitBccMesh[n];
#else
	m_meshes = new BccMesh[n];
#endif
	
	unsigned ntet = 0;
	unsigned nvert = 0;
    float vlm;
	unsigned i=0;
	for(;i<n;i++) {
#if WORLD_USE_FIT
        m_meshes[i].create(m_cluster->group(i), m_triIntersect);
#else
		m_meshes[i].create(m_cluster->group(i), m_anchorIntersect, 5);
#endif
        vlm = m_meshes[i].calculateVolume();
        m_meshes[i].setVolume(vlm);
		ntet += m_meshes[i].numTetrahedrons();
		nvert += m_meshes[i].numPoints();
	}
	
	std::cout<<"\n n tetrahedron meshes "<<n
	<<"\n total n tetrahedrons "<<ntet
	<<"\n total n points "<<nvert
	<<"\n";
	m_numMeshes = n;
	m_totalNumTetrahedrons = ntet;
	m_totalNumPoints = nvert;
}

void BccWorld::createTriangleMeshesFromFile()
{
	m_triangleMeshes = new GeometryArray;
	if(!readTriangleDataFromFile()) return;
	
	std::cout<<"\n n triangle mesh: "<<m_triangleMeshes->numGeometries()
	<<"\n";
	m_triIntersect = new KdIntersection;
	for(unsigned i=0; i<m_triangleMeshes->numGeometries(); i++) {
		ATriangleMesh * m = (ATriangleMesh *)m_triangleMeshes->geometry(i);
		std::cout<<m->dagName()
		<<"\n n triangle: "<<m->numTriangles()
		<<"\n n points: "<<m->numPoints()
		<<"\n";
		m_triIntersect->addGeometry(m);
	}
	
	KdTree::MaxBuildLevel = 32;
	KdTree::NumPrimitivesInLeafThreashold = 9;
	m_triIntersect->create();
}
 
void BccWorld::draw()
{
    // BoundingBox box;
    // m_grid->getBounding(box);
    
    // glColor3f(.21f, .21f, .21f);
    // m_drawer->boundingBox(box);
    
    // m_grid->draw(m_drawer, (unsigned *)m_mesh.m_anchorBuf->data());

	drawTetrahedronMesh();
	drawAnchor();
	drawTriangleMesh();
    
	glDisable(GL_DEPTH_TEST);
    // glColor3f(.59f, .02f, 0.f);
    // drawCurves();
	// drawCurveStars();
	
	const unsigned selectedCurveGrp = m_cluster->currentGroup();
	if(m_cluster->isGroupIdValid(selectedCurveGrp)) {
		m_drawer->setGroupColorLight(selectedCurveGrp);
		m_drawer->geometry(m_cluster->group(selectedCurveGrp));
	}
	
	// m_drawer->drawKdTree(m_triIntersect);
}

bool BccWorld::readCurveDataFromFile()
{
	if(BaseFile::InvalidFilename(BccGlobal::FileName)) 
		return false;
	
	if(!BaseFile::FileExists(BccGlobal::FileName)) {
		BccGlobal::FileName = "unknown";
		return false;
	}
	
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RCurve);
	hes.addCurve("curves", m_curves);
	if(!hes.open(BccGlobal::FileName)) return false;
	hes.close();
	
	return true;
}

bool BccWorld::readTriangleDataFromFile()
{
	if(BaseFile::InvalidFilename(BccGlobal::FileName)) 
		return false;
	
	if(!BaseFile::FileExists(BccGlobal::FileName)) {
		BccGlobal::FileName = "unknown";
		return false;
	}
	
	HesperisFile hes;
	hes.setReadComponent(HesperisFile::RTri);
	if(!hes.open(BccGlobal::FileName)) return false;
	hes.close();
	
	hes.extractTriangleMeshes(m_triangleMeshes);
	
	return true;
}

void BccWorld::drawTetrahedronMesh()
{
	unsigned i=0;
	for(;i<m_numMeshes; i++) {
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor3f(0.51f, 0.53f, 0.52f);
	drawTetrahedronMesh(m_meshes[i].numTetrahedrons(), m_meshes[i].points(),
	         m_meshes[i].indices());

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3f(.03f, .14f, .44f);
    drawTetrahedronMesh(m_meshes[i].numTetrahedrons(), m_meshes[i].points(),
	         m_meshes[i].indices());
	}
}

void BccWorld::drawTetrahedronMesh(unsigned nt, Vector3F * points, unsigned * indices)
{
    glBegin(GL_TRIANGLES);
    unsigned i, j;
    Vector3F q;
    unsigned * tet;
    for(i=0; i< nt; i++) {
        tet = &indices[i*4];
        for(j=0; j< 12; j++) {
            q = points[ tet[ TetrahedronToTriangleVertex[j] ] ];
            glVertex3fv((GLfloat *)&q);
        }
    }
    glEnd();
}

void BccWorld::drawAnchor()
{
    unsigned i, j;
    const float csz = BaseCurve::RayIntersectionTolerance;//m_grid->span() / 1024.f;
	// m_drawer->setWired(0);
	// unsigned nt = m_mesh.m_numTetrahedrons;
	glColor3f(.59f, .21f, 0.f);
	for(i=0; i<m_numMeshes;i++) {
	    Vector3F * p = (Vector3F *)m_meshes[i].points();
	    unsigned * a = (unsigned *)m_meshes[i].anchors();
	    for(j=0;j<m_meshes[i].numPoints();j++) {
	        if(a[j]<1) continue;
	        m_drawer->cube(p[j], csz);
	    }
    // unsigned * t = (unsigned *)m_mesh.m_indexBuf->data();
    // Vector3F q;
    
    //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    //glBegin(GL_TRIANGLES);
    // for(i=0; i< nt; i++) {
        // bool anchored = 1;
        // unsigned * tet = &t[i*4];
        // for(j=0; j<4; j++) {
            // if(a[tet[j]] < 1) {
                // anchored = 0;
                // break;
            // }
        // }
        // if(!anchored) continue;
        // for(j=0; j< 12; j++) {
            // q = p[ tet[ TetrahedronToTriangleVertex[j] ] ];
            // glVertex3fv((GLfloat *)&q);
        // }
    // }
    //glEnd();
    }
}

void BccWorld::drawTriangleMesh()
{
	glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor3f(0.63f, 0.64f, 0.65f);
	m_drawer->geometry(m_triangleMeshes);
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glColor3f(.03f, .14f, .44f);
    m_drawer->geometry(m_triangleMeshes);
}

bool BccWorld::save()
{
	if(BaseFile::InvalidFilename(BccGlobal::FileName)) {
		std::cout<<" no specifc file to save\n";
		return false;
	}
	
	HesperisFile hes;
	hes.setWriteComponent(HesperisFile::WTetra);
	
#if 0	
	unsigned i = 0;
	for(; i < m_numMeshes; i++) {
		std::stringstream sst;
		sst<<"tetra_"<<i;
		hes.addTetrahedron(sst.str(), &m_meshes[i]);
	}
#else
	ATetrahedronMesh * cm = combinedTetrahedronMesh();
	hes.addTetrahedron("tetra_c", cm);
#endif	

	if(!hes.open(BccGlobal::FileName)) return false;
	hes.setDirty();
	hes.save();
	hes.close();
	
#if 0
#else
	delete cm;
#endif

	return true;
}

const float BccWorld::totalCurveLength() const
{ return m_totalCurveLength; }

const unsigned BccWorld::numCurves() const
{ return m_numCurves; }

const unsigned BccWorld::numTetrahedrons() const
{ return m_totalNumTetrahedrons; }

const unsigned BccWorld::numPoints() const
{ return m_totalNumPoints; }

void BccWorld::clearTetrahedronMesh()
{
    if(m_meshes) delete[] m_meshes;
}

void BccWorld::rebuildTetrahedronsMesh(float deltaNumGroups)
{
    m_estimatedNumGroups += deltaNumGroups * numCurves();
    if(m_estimatedNumGroups < 100.f) m_estimatedNumGroups = 100.f;
    
    clearTetrahedronMesh();
    createTetrahedronMeshes();
    std::cout<<" done rebuild. \n";
}

void BccWorld::select(const Ray * r)
{
	BaseCurve::RayIntersectionTolerance = totalCurveLength() / m_estimatedNumGroups * .37f;
	if(!m_cluster->intersectRay(r))
		clearSelection();
}

void BccWorld::clearSelection()
{ m_cluster->setCurrentGroup(m_cluster->numGroups()); }

float BccWorld::groupCurveLength(GeometryArray * geos)
{
	float sum = 0.f;
	const unsigned n = geos->numGeometries();
     unsigned i = 0;
    for(;i<n;i++) {
        BezierCurve * c = static_cast<BezierCurve *>(geos->geometry(i));
        sum += c->length();
    }
	return sum;
}

void BccWorld::reduceSelected(float x)
{
    const unsigned selectedCurveGrp = m_cluster->currentGroup();
	if(m_cluster->isGroupIdValid(selectedCurveGrp))
		reduceGroup(selectedCurveGrp);
	else
        reduceAllGroups();	
}

void BccWorld::reduceAllGroups()
{
	unsigned i = 0;
	for(;i<m_cluster->numGroups();i++) reduceGroup(i);
}

void BccWorld::reduceGroup(unsigned igroup)
{
	const float oldCurveLength = groupCurveLength(m_cluster->group(igroup));
	const unsigned oldNCurves = m_cluster->group(igroup)->numGeometries();
    GeometryArray * reduced = 0;
	
	int i=0;
	for(;i<20;i++) {
		GeometryArray * ir = m_reducer->compute(m_cluster->group(igroup), FitBccMeshBuilder::EstimatedGroupSize);
		if(ir) {
			reduced = ir;
			m_cluster->setGroupGeometry(igroup, reduced);
		}
		else break;
	}
	
	if(!reduced) {
        std::cout<<" bcc has insufficient for curve reduction, skipped.\n";
        return;
    }
	
	m_totalCurveLength -= oldCurveLength;
	m_totalCurveLength += groupCurveLength(reduced);
	
	m_numCurves -= oldNCurves;
	m_numCurves += reduced->numGeometries();
	
	rebuildGroupTetrahedronMesh(igroup, reduced);
}

void BccWorld::rebuildGroupTetrahedronMesh(unsigned igroup, GeometryArray * geos)
{
	const unsigned oldNVert = m_meshes[igroup].numPoints();
	const unsigned oldNTet = m_meshes[igroup].numTetrahedrons();
	const unsigned oldTotalNTet = m_totalNumTetrahedrons;
	const unsigned oldTotalNVert = m_totalNumPoints;
	float vlm;
#if WORLD_USE_FIT
	m_meshes[igroup].create(geos, m_triIntersect);
#else
	m_meshes[igroup].create(geos, m_anchorIntersect, 5);
#endif	
	vlm = m_meshes[igroup].calculateVolume();
	m_meshes[igroup].setVolume(vlm);
	
	m_totalNumPoints -= oldNVert;
	m_totalNumPoints += m_meshes[igroup].numPoints();
	m_totalNumTetrahedrons -= oldNTet;
	m_totalNumTetrahedrons += m_meshes[igroup].numTetrahedrons();
	
	std::cout<<" reduce n points from "<<oldTotalNTet<<" to "<<m_totalNumPoints
	<<"\n n tetrahedrons form "<<oldTotalNTet<<" to "<<m_totalNumTetrahedrons
	<<"\n";
}

ATetrahedronMesh * BccWorld::combinedTetrahedronMesh()
{
	ATetrahedronMesh * omesh = new ATetrahedronMesh;
	unsigned ntet = 0;
	unsigned nvert = 0;
	unsigned i = 0;
	for(; i < m_numMeshes; i++) {
		ntet += m_meshes[i].numTetrahedrons();
		nvert += m_meshes[i].numPoints();
	}
	omesh->create(nvert, ntet);
	
	ntet = 0;
	nvert = 0;
	i = 0;
	for(; i < m_numMeshes; i++) {
		omesh->copyStripe(&m_meshes[i], nvert, ntet * 4);
		ntet += m_meshes[i].numTetrahedrons();
		nvert += m_meshes[i].numPoints();
	}
	float vlm = omesh->calculateVolume();
	omesh->setVolume(vlm);
	std::cout<<" combined all meshes:\n n tetrahedrons "<<ntet
	<<"\n n points "<<nvert
	<<"\n initial volume "<<vlm
	<<"\n";
	return omesh;
}
//:~