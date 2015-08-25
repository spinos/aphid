#include "BccWorld.h"
#include <BezierCurve.h>
#include <KdTreeDrawer.h>
#include <CurveGroup.h>
#include <HesperisFile.h>
#include <KdCluster.h>
#include <KdIntersection.h>
#include <GeometryArray.h>
#include <CurveBuilder.h>
#include <BezierCurve.h>
#include <RandomCurve.h>
#include <bezierPatch.h>
#include <ATriangleMesh.h>
#include "FitBccMeshBuilder.h"
#include "CurveReduction.h"
#include <ATetrahedronMeshGroup.h>
#include <BlockBccMeshBuilder.h>

BccWorld::BccWorld()
{
    m_numCurves = 0;
	m_totalCurveLength = 0.f;
    m_estimatedNumGroups = 2500.f;
	m_triangleMeshes = NULL;
	m_reducer = new CurveReduction;
	m_blockBuilder = new BlockBccMeshBuilder;
	m_fitBuilder = new FitBccMeshBuilder;
	m_curveCluster = NULL;
	m_patchCluster = NULL;
}

BccWorld::~BccWorld() 
{
    delete m_reducer;
	delete m_curveCluster;
	delete m_patchCluster;
	delete m_triangleMeshes;
	delete m_blockBuilder;
	delete m_fitBuilder;
}

void BccWorld::setTiangleGeometry(GeometryArray * x)
{ m_triangleMeshes = x; }

void BccWorld::addCurveGroup(CurveGroup * m)
{ m_curveGeos.push_back(m); }

GeometryArray * BccWorld::selectedGroup(unsigned & idx) const
{
	if(!m_curveCluster) return NULL;
	idx = m_curveCluster->currentGroup();
	if(!m_curveCluster->isGroupIdValid(idx)) return NULL;
	return m_curveCluster->group(idx);
}

float BccWorld::drawAnchorSize() const
{ return FitBccMeshBuilder::EstimatedGroupSize / 6.f; }

const float BccWorld::totalCurveLength() const
{ return m_totalCurveLength; }

const unsigned BccWorld::numCurves() const
{ return m_numCurves; }

const unsigned BccWorld::numTetrahedrons() const
{ return m_totalNumTetrahedrons; }

const unsigned BccWorld::numPoints() const
{ return m_totalNumPoints; }

ATetrahedronMesh * BccWorld::tetrahedronMesh(unsigned i)
{ return m_tetrahedonMeshes[i]; }

void BccWorld::clearTetrahedronMesh()
{ 
	std::vector<ATetrahedronMeshGroup *>::iterator it = m_tetrahedonMeshes.begin();
	for(;it!=m_tetrahedonMeshes.end();++it) delete *it;
	m_tetrahedonMeshes.clear(); 
}

unsigned BccWorld::numTetrahedronMeshes() const
{ return m_tetrahedonMeshes.size(); }

GeometryArray * BccWorld::triangleGeometries() const
{ return m_triangleMeshes; }

unsigned BccWorld::numTriangles() const
{
	if(!m_triangleMeshes) return 0;
	const unsigned m = m_triangleMeshes->numGeometries();
	unsigned nt = 0;
	unsigned i = 0;
	for(;i<m;i++) nt += ((ATriangleMesh *)m_triangleMeshes->geometry(i))->numTriangles();
	return nt;
}

void BccWorld::rebuildTetrahedronsMesh(float deltaNumGroups)
{
    m_estimatedNumGroups += deltaNumGroups * numCurves();
    if(m_estimatedNumGroups < 100.f) m_estimatedNumGroups = 100.f;
    
    clearTetrahedronMesh();
    createTetrahedronMeshesByFitCurves();
	createTetrahedronMeshesByBlocks();
    std::cout<<" done rebuild. \n";
}

void BccWorld::select(const Ray * r)
{
	if(!m_curveCluster) return;
	BaseCurve::RayIntersectionTolerance = totalCurveLength() / m_estimatedNumGroups * .37f;
	if(!m_curveCluster->intersectRay(r))
		clearSelection();
}

void BccWorld::clearSelection()
{ m_curveCluster->setCurrentGroup(m_curveCluster->numGroups()); }

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
    const unsigned selectedCurveGrp = m_curveCluster->currentGroup();
	if(m_curveCluster->isGroupIdValid(selectedCurveGrp))
		reduceGroup(selectedCurveGrp);
	else
        reduceAllGroups();	
}

void BccWorld::reduceAllGroups()
{
	unsigned i = 0;
	for(;i<m_curveCluster->numGroups();i++) reduceGroup(i);
}

void BccWorld::reduceGroup(unsigned igroup)
{
	const float oldCurveLength = groupCurveLength(m_curveCluster->group(igroup));
	const unsigned oldNCurves = m_curveCluster->group(igroup)->numGeometries();
    GeometryArray * reduced = 0;
	
	int i=0;
	for(;i<20;i++) {
		GeometryArray * ir = m_reducer->compute(m_curveCluster->group(igroup), FitBccMeshBuilder::EstimatedGroupSize);
		if(ir) {
			reduced = ir;
			m_curveCluster->setGroupGeometry(igroup, reduced);
		}
		else break;
	}
	
	if(!reduced) {
        std::cout<<" insufficient condition for curve reduction, skipped.\n";
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
	ATetrahedronMeshGroup * amesh = m_tetrahedonMeshes[igroup];
	const unsigned oldNVert = amesh->numPoints();
	const unsigned oldNTet = amesh->numTetrahedrons();
	const unsigned oldTotalNTet = m_totalNumTetrahedrons;
	const unsigned oldTotalNVert = m_totalNumPoints;
	
	delete amesh;
	
	m_tetrahedonMeshes[igroup] = genTetFromGeometry(geos, m_fitBuilder);
	
	m_totalNumPoints -= oldNVert;
	m_totalNumPoints += m_tetrahedonMeshes[igroup]->numPoints();
	m_totalNumTetrahedrons -= oldNTet;
	m_totalNumTetrahedrons += m_tetrahedonMeshes[igroup]->numTetrahedrons();
	
	std::cout<<" reduce n points from "<<oldTotalNTet<<" to "<<m_totalNumPoints
	<<"\n n tetrahedrons form "<<oldTotalNTet<<" to "<<m_totalNumTetrahedrons
	<<"\n";
}

ATetrahedronMeshGroup * BccWorld::combinedTetrahedronMesh()
{
	ATetrahedronMeshGroup * omesh = new ATetrahedronMeshGroup;
	unsigned ntet = 0;
	unsigned nvert = 0;
    unsigned nstrip = 0;
	const unsigned n = numTetrahedronMeshes();
	unsigned i = 0;
	for(; i < n; i++) {
		ATetrahedronMeshGroup * imesh = m_tetrahedonMeshes[i];
		ntet += imesh->numTetrahedrons();
		nvert += imesh->numPoints();
        nstrip += imesh->numStripes();
	}
	omesh->create(nvert, ntet, nstrip);
 
	ntet = 0;
	nvert = 0;
    nstrip = 0;
	i = 0;
	for(; i < n; i++) {
		ATetrahedronMeshGroup * imesh = m_tetrahedonMeshes[i];
        omesh->copyPointDrift(imesh->pointDrifts(), 
                              imesh->numStripes(), 
                              nstrip,
                              nvert);
        omesh->copyIndexDrift(imesh->indexDrifts(), 
                              imesh->numStripes(), 
                              nstrip,
                              ntet*4);
        
		omesh->copyStripe(imesh, nvert, ntet * 4);
		ntet += imesh->numTetrahedrons();
		nvert += imesh->numPoints();
        nstrip += imesh->numStripes();
	}
	float vlm = omesh->calculateVolume();
	omesh->setVolume(vlm);
	std::cout<<" combined all meshes:\n n tetrahedrons "<<ntet
	<<"\n n points "<<nvert
	<<"\n initial volume "<<vlm
	<<"\n";
    omesh->verbose();
	return omesh;
}

void BccWorld::addCurveGeometriesToCluster(CurveGroup * data)
{
	unsigned * cc = data->counts();
	Vector3F * cvs = data->points();
	const unsigned n = data->numCurves();
	
	GeometryArray * geos = new GeometryArray;
	geos->create(n);
	
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
		
		geos->setGeometry(c, i);
		
		cvDrift += ncv;
	}
	
	m_curveCluster->addGeometry(geos);
}

bool BccWorld::createTriangleIntersection()
{	
	if(!m_triangleMeshes) return false;
	unsigned i;
	m_triIntersect = new KdIntersection;
	for(i=0; i<m_triangleMeshes->numGeometries(); i++) {
		ATriangleMesh * m = (ATriangleMesh *)m_triangleMeshes->geometry(i);
		std::cout<<"\n mesh["<<i<<"] n triangles: "<<m->numTriangles()
		<<"\n n points: "<<m->numPoints()
		<<"\n";
		m_triIntersect->addGeometry(m);
	}
	
	KdTree::MaxBuildLevel = 20;
	KdTree::NumPrimitivesInLeafThreashold = 9;
	m_triIntersect->create();
	return true;
}

bool BccWorld::buildTetrahedronMesh()
{	
	if(!createTriangleIntersection()) {
		std::cout<<"\n bcc world has no grow mesh ";
		return false;
	}
	
	createTetrahedronMeshesByFitCurves();
	createTetrahedronMeshesByBlocks();
	return true;
}
	
bool BccWorld::createCurveCluster()
{
	const unsigned n = m_curveGeos.size();
	if(n < 1) return false;
	
	std::cout<<"\n bcc world creating curve cluster ";
	if(m_curveCluster) delete m_curveCluster;
	m_curveCluster = new KdCluster;
	
	m_totalCurveLength = 0.f;
	m_numCurves = 0;
	
	unsigned i;
	for(i=0; i< n; i++) {
		addCurveGeometriesToCluster(m_curveGeos[i]);
		m_numCurves += m_curveGeos[i]->numCurves();
	}
	
	std::cout<<"\n total n curves "<<m_numCurves
	    <<" total curve length: "<<m_totalCurveLength<<"\n";
	
	KdTree::MaxBuildLevel = 6;
	KdTree::NumPrimitivesInLeafThreashold = 31;
	
	m_curveCluster->create();
	return true;
}

void BccWorld::addPatchBoxes(const std::vector<AOrientedBox> & src)
{
	std::vector<AOrientedBox>::const_iterator it = src.begin();
	for(;it!=src.end();++it) m_patchBoxes.push_back(*it);
}

const std::vector<AOrientedBox> * BccWorld::patchBoxes() const
{ return &m_patchBoxes; }

void BccWorld::createTetrahedronMeshesByFitCurves()
{
    if(!createCurveCluster()) {
        std::cout<<"\n no curve available in bcc world!";
        return;
    }
	
	std::cout<<"\n bcc world building tetrahedron mesh along curve ";

    FitBccMeshBuilder::EstimatedGroupSize = totalCurveLength() / m_estimatedNumGroups;
    std::cout<<"\n estimate group size "<<FitBccMeshBuilder::EstimatedGroupSize;
	
	unsigned n = m_curveCluster->numGroups();
	
	unsigned i;
	for(i=0;i<n;i++)
		m_tetrahedonMeshes.push_back(genTetFromGeometry(m_curveCluster->group(i), m_fitBuilder));
		
	unsigned ntet = 0;
	unsigned nvert = 0;
	unsigned nanchored = 0;
	unsigned nstripes = 0;
	for(i=0;i<n;i++) {
        ATetrahedronMeshGroup * amesh = m_tetrahedonMeshes[i];
		ntet += amesh->numTetrahedrons();
		nvert += amesh->numPoints();
		nanchored += amesh->numAnchoredPoints();
        nstripes += amesh->numStripes();
	}
	
	std::cout<<"\n n tetrahedron meshes "<<n
	<<"\n total n tetrahedrons "<<ntet
	<<"\n total n points "<<nvert
	<<"\n total n anchored points "<<nanchored
    <<"\n total n stripes "<<nstripes
	<<"\n";
	m_totalNumTetrahedrons = ntet;
	m_totalNumPoints = nvert;
}

ATetrahedronMeshGroup * BccWorld::genTetFromGeometry(GeometryArray * geos, TetrahedronMeshBuilder * builder)
{
	unsigned ntet, nvert, nstripes;
	
	builder->build(geos, ntet, nvert, nstripes);
	
	ATetrahedronMeshGroup * amesh = new ATetrahedronMeshGroup;
	amesh->create(nvert, ntet, nstripes);
	
	builder->getResult(amesh);
	amesh->checkTetrahedronVolume();
	amesh->clearAnchors();
	
	builder->addAnchors(amesh, nstripes, m_triIntersect);
	
	float vlm = amesh->calculateVolume();
	amesh->setVolume(vlm);
	
	return amesh;
}

bool BccWorld::createPatchCluster()
{
	const unsigned n = m_patchBoxes.size();
	if(n < 1) return false;
	
	std::cout<<"\n bcc world creating patch cluster ";
	if(m_patchCluster) delete m_patchCluster;
	m_patchCluster = new KdCluster;
	
	GeometryArray * geos = new GeometryArray;
	geos->create(n);
	
	float totalPatchLength = 0.f;
	
	unsigned i;
	for(i=0; i< n; i++) {
		totalPatchLength += m_patchBoxes[i].extent().x * 2.f;
		geos->setGeometry(&m_patchBoxes[i], i);
	}
	
	m_patchCluster->addGeometry(geos);
	
	std::cout<<"\n total n patches "<<n
	    <<" total patch length: "<<totalPatchLength<<"\n";
	
	KdTree::MaxBuildLevel = 6;
	KdTree::NumPrimitivesInLeafThreashold = 31;
	
	m_patchCluster->create();
	
	return true;
}

void BccWorld::createTetrahedronMeshesByBlocks()
{
	if(!createPatchCluster()) {
        std::cout<<"\n no patch available in bcc world!";
        return;
    }

	unsigned n = m_patchCluster->numGroups();
	unsigned i;
	for(i=0;i<n;i++)
		m_tetrahedonMeshes.push_back(genTetFromGeometry(m_patchCluster->group(i), m_blockBuilder));
	
}
//:~