/*
 *  rodt
 */
#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include <math/BaseCamera.h>
#include "glwidget.h"
#include <SolverThread.h>
#include <pbd/WindTurbine.h>
#include <ogl/RotationHandle.h>
#include <ttg/GenericHexahedronGrid.h>
#include <smp/UniformGrid8Sphere.h>

using namespace aphid;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	perspCamera()->setFarClipPlane(1000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(1000.f);
	orthoCamera()->setNearClipPlane(1.f);
	//usePerspCamera();
	//resetView();
	m_workMode = wmInteractive;
	m_smpV.set(.3f, .4f, .5f);
	m_solver = new SolverThread;
	
	pbd::WindTurbine* windicator = m_solver->windTurbine();
	m_roth = new RotationHandle(windicator->visualizeSpace() );
	m_roth->setRadius(8.f);
	
	m_grd = new GridTyp;
	int ndiv = 8;
	float indiv = 2.f / (float)ndiv;
	float nh = ndiv / 2.f;
	int np = (ndiv+1) * (ndiv+1) * (ndiv+1);
	int nc = ndiv * ndiv * ndiv;
	m_grd->create(np, nc);
	
	int acc = 0;
	for(int k=0;k<=ndiv;++k) {
		for(int j=0;j<=ndiv;++j) {
			for(int i=0;i<=ndiv;++i) {
				m_grd->setPos(Vector3F(1.f * i - nh, 1.f * j - nh, 1.f * k - nh) * indiv, acc++);
			}
		}
	}
	
	int nppj = ndiv + 1;
	int nppk = nppj * nppj;
	int cellVert[8];
	acc = 0;
	for(int k=0;k<ndiv;++k) {
		int k1 = k + 1;
		for(int j=0;j<ndiv;++j) {
			int j1 = j + 1;
			for(int i=0;i<ndiv;++i) {
				int i1 = i + 1;
				
				cellVert[0] = i + j * nppj + k * nppk;
				cellVert[1] = i1 + j * nppj + k * nppk;
				cellVert[2] = i + j1 * nppj + k * nppk;
				cellVert[3] = i1 + j1 * nppj + k * nppk;
				cellVert[4] = i + j * nppj + k1 * nppk;
				cellVert[5] = i1 + j * nppj + k1 * nppk;
				cellVert[6] = i + j1 * nppj + k1 * nppk;
				cellVert[7] = i1 + j1 * nppj + k1 * nppk;
				
				m_grd->setCell(cellVert, acc++);
			}
		}
	}
	
	int inds[512];
	int ns = 0;
	cvx::Hexahedron hexa;
	for(int i=0;i<m_grd->numCells();++i) {
		m_grd->getCell(hexa, i);
		Vector3F dp = hexa.getCenter();
		if(dp.length() < 1.f) {			
			inds[i] = ns;
			ns++;
			
		} else {
			inds[i] = -1;
		}
	}
	
	for(int i=0;i<m_grd->numCells();++i) {
		if(inds[i] < 0) {
/// put into sphere
			int iz = i / 64;
			if(iz > 7) iz = 7;
			int iy = (i - iz * 64) / 8;
			if(iy > 7) iy = 7;
			if(iy < 0) iy = 0;
			int ix = i - iz * 64 - iy * 8;
			if(ix > 7) ix = 7;
			if(ix < 0) ix = 0;
			Vector3F p1(-.875f + .25f * ix,
						-.875f + .25f * iy,
						-.875f + .25f * iz);
			p1.normalize();
			p1 *= .99f;
			
			inds[i] = smp::UniformGrid8Sphere::GetSampleInd(p1);
		}
#if 0
		std::cout<<" "<<inds[i]<<",";
		if(((i+1)&63) == 0) std::cout<<"\n";
#endif
	}
	std::cout.flush();
	
	qDebug()<<" grd np "<<m_grd->numPoints()
		<<" nc "<<m_grd->numCells()
		<<" nsample "<<ns;
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
	connect(this, SIGNAL(sendBeginCache()), m_solver, SLOT(recvBeginCache()) );
	connect(m_solver, SIGNAL(doneCache()), this, SLOT(recvEndCache()) );
	
}

void GLWidget::clientDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	const pbd::ParticleData* particle = m_solver->c_particles();
	const Vector3F * pos = particle->pos();
	const pbd::ParticleData* ghost = m_solver->c_ghostParticles();
	const Vector3F * gpos = ghost->pos();
	
	const int ne = m_solver->numEdges();
	int iA, iB, iG;
	glBegin(GL_LINES);
	for(int i=0; i< ne;++i) {
		m_solver->getEdgeIndices(iA, iB, iG, i);
	    const Vector3F& p1 = pos[iA];
	    const Vector3F& p2 = pos[iB];
	    const Vector3F& p3 = gpos[iG];
		glColor3f(1,1,1);
	    glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
		glColor3f(0,1,0);
		glVertex3f((p1.x + p2.x) * .5f,
		            (p1.y + p2.y) * .5f,
		            (p1.z + p2.z) * .5f);
		glVertex3f(p3.x,p3.y,p3.z);
	}
	glEnd();
	
	const int& ngp = ghost->numParticles();
	glColor3f(1,1,0);
	glBegin(GL_POINTS);
	for(int i=0; i< ngp;++i) {
		const Vector3F& p1 = gpos[i];
		glVertex3f(p1.x,p1.y,p1.z);
	}
	glEnd();

	drawWindTurbine();
	
	getDrawer()->m_markerProfile.apply();
#if 0
	drawGrid();
#endif
	Vector3F veye = getCamera()->eyeDirection();
	
	pbd::WindTurbine* windicator = m_solver->windTurbine();
	Matrix44F meye = *windicator->visualizeSpace();
	
	meye.setFrontOrientation(veye );

	m_roth->draw(&meye);
	
}

void GLWidget::clientSelect(QMouseEvent *event)
{
    if(m_workMode > wmInteractive) return;
	m_roth->begin(getIncidentRay() );
    //m_tranh->begin(getIncidentRay() );
	update();
}

void GLWidget::clientDeselect(QMouseEvent *event)
{
    if(m_workMode > wmInteractive) return;
	m_roth->end();
    //m_tranh->end();
	update();
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
    if(m_workMode > wmInteractive) return;
	m_roth->rotate(getIncidentRay() );
    //m_tranh->translate(getIncidentRay() );
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	switch (e->key()) {
		case Qt::Key_N:
			addWindSpeed(-5.f * RandomF01());
			break;
		case Qt::Key_M:
			addWindSpeed(5.f * RandomF01());
			break;
		case Qt::Key_V:
			shuffleSample();
			break;
		case Qt::Key_C:
			makeDynCache();
			break;
		default:
			break;
	}

	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

void GLWidget::resetPerspViewTransform()
{
static const float mm[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					32.f, 200.f, 346.4101616f, 1.f};
	Matrix44F mat(mm);
	perspCamera()->setViewTransform(mat, 400.f);
}

void GLWidget::resetOrthoViewTransform()
{
static const float mm1[16] = {1.f, 0.f, 0.f, 0.f,
					0.f, 0.8660254f, -0.5f, 0.f,
					0.f, 0.5f, 0.8660254f, 0.f,
					32.f, 200.f, 346.4101616f, 1.f};
	Matrix44F mat(mm1);
	orthoCamera()->setViewTransform(mat, 400.f);
	orthoCamera()->setHorizontalAperture(150.f);
}

void GLWidget::drawWindTurbine()
{
	const pbd::WindTurbine* windicator = m_solver->windTurbine();
	const Matrix44F* tm = windicator->visualizeSpace();
	
	getDrawer()->m_surfaceProfile.apply();
	
	glPushMatrix();
	getDrawer()->useSpace(*tm);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	
	drawMesh(pbd::WindTurbine::sStatorNumTriangleIndices, pbd::WindTurbine::sStatorMeshTriangleIndices,
		pbd::WindTurbine::sStatorMeshVertices, pbd::WindTurbine::sStatorMeshNormals);
	
	glRotatef(windicator->rotorAngle(), 1, 0, 0);
	
	drawMesh(pbd::WindTurbine::sRotorNumTriangleIndices, pbd::WindTurbine::sRotorMeshTriangleIndices,
		pbd::WindTurbine::sRotorMeshVertices, pbd::WindTurbine::sRotorMeshNormals);
		
	drawMesh(pbd::WindTurbine::sBladeNumTriangleIndices, pbd::WindTurbine::sBladeMeshTriangleIndices,
		pbd::WindTurbine::sBladeMeshVertices, pbd::WindTurbine::sBladeMeshNormals);
		
	glRotatef(120.f, 1, 0, 0);
	
	drawMesh(pbd::WindTurbine::sBladeNumTriangleIndices, pbd::WindTurbine::sBladeMeshTriangleIndices,
		pbd::WindTurbine::sBladeMeshVertices, pbd::WindTurbine::sBladeMeshNormals);
	
	glRotatef(120.f, 1, 0, 0);
	
	drawMesh(pbd::WindTurbine::sBladeNumTriangleIndices, pbd::WindTurbine::sBladeMeshTriangleIndices,
		pbd::WindTurbine::sBladeMeshVertices, pbd::WindTurbine::sBladeMeshNormals);
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopMatrix();
	
}

void GLWidget::drawMesh(const int& nind, const int* inds, const float* pos, const float* nml)
{
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)nml);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)pos);
	
	glDrawElements(GL_TRIANGLES, nind, GL_UNSIGNED_INT, inds );
	
}

void GLWidget::addWindSpeed(float x)
{
	pbd::WindTurbine* windicator = m_solver->windTurbine();
	float s = x + windicator->windSpeed();
	windicator->setWindSpeed(s);
}

void GLWidget::drawGrid()
{
	glPushMatrix();
	glScalef(40,40,40);
	glColor3f(0.2f,1.f,1.f);
	cvx::Hexahedron hexa;
	
	glBegin(GL_POINTS);
	for(int i=0;i<smp::UniformGrid8Sphere::sNumSamples;++i) {
		glVertex3fv(smp::UniformGrid8Sphere::sSamplePnts[i]);
	}
	glEnd();
	
	glColor3f(0.9f,0.f,0.f);
	getDrawer()->arrow(Vector3F(0.f, 0.f, 0.f), m_smpV);
	
	const Vector3F vsmp = smp::UniformGrid8Sphere::GetSamplePnt(m_smpV);
	glColor3f(0.9f,.9f,0.f);
	getDrawer()->arrow(Vector3F(0.f, 0.f, 0.f), vsmp);
	
	glPopMatrix();
}

void GLWidget::shuffleSample()
{	
	m_smpV.set(RandomFn11(), RandomFn11(), RandomFn11() );
	if(m_smpV.length() > 1.f) m_smpV.normalize();
	update();
}

void GLWidget::makeDynCache()
{
    pbd::WindTurbine* windicator = m_solver->windTurbine();
	const float& s = windicator->windSpeed();
	if(s < 4.f) {
	    qDebug()<<" wind speed "<<s<<" is too low, skip making cache";
	    return;
	}
	qDebug()<<" cache wind speed "<<s;
	m_solver->setCacheWindSpeed(s);
	beginCaching();
	
}

void GLWidget::beginCaching()
{
    m_workMode = wmMakeingCache;
    emit sendBeginCache();
}

void GLWidget::recvEndCache()
{ 
    qDebug()<<" end cache wind ";
    m_workMode = wmInteractive; 
}
