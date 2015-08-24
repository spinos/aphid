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
#include <ATriangleMesh.h>
#include "FitBccMeshBuilder.h"
#include "CurveReduction.h"
#include <ATetrahedronMeshGroup.h>
#include <BlockBccMeshBuilder.h>
#include <FitBccMeshBuilder.h>

BccWorld::BccWorld()
{
    m_numCurves = 0;
	m_numMeshes = 0;
    m_totalCurveLength = 0.f;
    m_estimatedNumGroups = 2500.f;
	m_triangleMeshes = NULL;
	m_reducer = new CurveReduction;
	m_blockBuilder = new BlockBccMeshBuilder;
	m_fitBuilder = new FitBccMeshBuilder;
	m_cluster = 0;
}

BccWorld::~BccWorld() 
{
    delete m_reducer;
	delete m_cluster;
	delete m_allGeo;
	// delete[] m_meshes;
	delete m_triangleMeshes;
	delete m_blockBuilder;
	delete m_fitBuilder;
}

void BccWorld::setTiangleGeometry(GeometryArray * x)
{ 
	m_triangleMeshes = x;
	createTriangleIntersection();
}

void BccWorld::addCurveGroup(CurveGroup * m)
{ m_curveGeos.push_back(m); }

void BccWorld::createTetrahedronMeshes()
{
	createTetrahedronMeshesByFitCurves();
	createTetrahedronMeshesByBlocks();
}

GeometryArray * BccWorld::selectedGroup(unsigned & idx) const
{
	if(!m_cluster) return NULL;
	idx = m_cluster->currentGroup();
	if(!m_cluster->isGroupIdValid(idx)) return NULL;
	return m_cluster->group(idx);
}

float BccWorld::drawAnchorSize() const
{ return FitBccMeshBuilder::EstimatedGroupSize / 8.f; }

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
{ return m_numMeshes; }

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
    createTetrahedronMeshes();
    std::cout<<" done rebuild. \n";
}

void BccWorld::select(const Ray * r)
{
	if(!m_cluster) return;
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
	
	m_tetrahedonMeshes[igroup] = fitAGroup(geos);
	
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
	unsigned i = 0;
	for(; i < m_numMeshes; i++) {
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
	for(; i < m_numMeshes; i++) {
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

bool BccWorld::createAllCurveGeometry()
{
	const unsigned n = m_curveGeos.size();
	if(n < 1) return false;
	unsigned i;
	
	unsigned m = 0;
	for(i=0; i< n; i++)
		m+= m_curveGeos[i]->numCurves();

	m_allGeo = new GeometryArray;
	m_allGeo->create(m);
	
	m_totalCurveLength = 0.f;
	
	unsigned geoDrift = 0;
	for(i=0; i< n; i++) {
		createCurveGeometry(geoDrift, m_curveGeos[i]);
		geoDrift += m_curveGeos[i]->numCurves();
	}
		
	std::cout<<" n curves "<<m
	    <<" total curve length: "<<m_totalCurveLength<<"\n";
    m_numCurves = m;
	return true;
}

void BccWorld::createCurveGeometry(unsigned geoBegin, CurveGroup * data)
{
	unsigned * cc = data->counts();
	Vector3F * cvs = data->points();
	const unsigned n = data->numCurves();
	
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
		
		m_allGeo->setGeometry(c, i + geoBegin);
		
		cvDrift += ncv;
	}
}

bool BccWorld::createTriangleIntersection()
{	
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
	if(!m_triIntersect) {
		std::cout<<"\n bcc world has no grow mesh ";
		return false;
	}
	if(!createAllCurveGeometry()) return false;
	
	std::cout<<"\n bcc world creating curve cluster ";
	m_cluster = new KdCluster;
	m_cluster->addGeometry(m_allGeo);
	
	KdTree::MaxBuildLevel = 6;
	KdTree::NumPrimitivesInLeafThreashold = 31;
	
	m_cluster->create();
	
	std::cout<<"\n bcc world building tetrahedron mesh ";
	createTetrahedronMeshes();
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
    if(totalCurveLength() < 1.f) {
        std::cout<<" invalid total curve length "<<totalCurveLength()<<" !\n";
        return;
    }

    FitBccMeshBuilder::EstimatedGroupSize = totalCurveLength() / m_estimatedNumGroups;
    std::cout<<"\n estimate group size "<<FitBccMeshBuilder::EstimatedGroupSize;
	
	unsigned n = m_cluster->numGroups();
	
	unsigned i;
	for(i=0;i<n;i++)
		m_tetrahedonMeshes.push_back(fitAGroup(m_cluster->group(i)));
	
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
	m_numMeshes = n;
	m_totalNumTetrahedrons = ntet;
	m_totalNumPoints = nvert;
}

ATetrahedronMeshGroup * BccWorld::fitAGroup(GeometryArray * geos)
{
	std::vector<Vector3F > tetrahedronP;
	std::vector<unsigned > tetrahedronInd;
    std::vector<unsigned > pdrifts;
    std::vector<unsigned > idrifts;
	
	m_fitBuilder->build(geos, tetrahedronP, tetrahedronInd,
                  pdrifts, idrifts);
				  
	unsigned ntet = tetrahedronInd.size()/4;
	unsigned nvert = tetrahedronP.size();
	unsigned nstripes = geos->numGeometries();
		
	ATetrahedronMeshGroup * amesh = new ATetrahedronMeshGroup;
	amesh->create(nvert, ntet, nstripes);
		
	unsigned i;
	for(i=0;i<nvert;i++) amesh->points()[i] = tetrahedronP[i];
	for(i=0;i<ntet*4;i++) amesh->indices()[i] = tetrahedronInd[i];
	for(i=0;i<nstripes;i++) amesh->pointDrifts()[i] = pdrifts[i];
	for(i=0;i<nstripes;i++) amesh->indexDrifts()[i] = idrifts[i];
	
	amesh->clearAnchors();
	
	m_fitBuilder->addAnchors(amesh, nstripes, m_triIntersect);
	
	tetrahedronP.clear();
	tetrahedronInd.clear();
	pdrifts.clear();
	idrifts.clear();
	
	float vlm = amesh->calculateVolume();
	amesh->setVolume(vlm);
	
	return amesh;
}

void BccWorld::createTetrahedronMeshesByBlocks()
{
// todo
}
//:~