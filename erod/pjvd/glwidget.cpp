/*
 *  projective rod
 */
#include <GeoDrawer.h>
#include <QtGui>
#include <QtOpenGL>
#include <math/BaseCamera.h>
#include "glwidget.h"
#include <TestSolver.h>
#include <pbd/WindTurbine.h>
#include <pbd/ShapeMatchingRegion.h>
#include <ogl/RotationHandle.h>

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
	m_solver = new TestSolver;
	
	pbd::WindTurbine* windicator = m_solver->windTurbine();
	m_roth = new RotationHandle(windicator->visualizeSpace() );
	m_roth->setRadius(8.f);
	
	std::cout.flush();
	
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    connect(internalTimer(), SIGNAL(timeout()), m_solver, SLOT(simulate()));
	connect(this, SIGNAL(restartAtCurrentState()), m_solver, SLOT(recvRestartAtCurrentState()));
	connect(this, SIGNAL(pauseSim()), m_solver, SLOT(recvToggleSimulation()));
	connect(m_solver, SIGNAL(doneStep()), this, SLOT(update()));
}

void GLWidget::clientDraw()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	const pbd::ParticleData* particle = m_solver->c_particles();
	const Vector3F * pos = particle->pos();
	glColor3f(0,0,1);
/*	
	const int& np = particle->numParticles();
	glBegin(GL_LINES);
	for(int i=1; i< np;++i) {
		const Vector3F& p1 = pos[i - 1];
		glVertex3f(p1.x,p1.y,p1.z);
		const Vector3F& p2 = pos[i];
		glVertex3f(p2.x,p2.y,p2.z);
	}
	glEnd();
*/
	const int nr = m_solver->numRegions();
	int v1, v2;
	glBegin(GL_LINES);
	for(int i=0; i< nr;++i) {
		const pbd::ShapeMatchingRegion* ri = m_solver->region(i);
		const int& ne = ri->numEdges();
		for(int j=0;j<ne;++j) {
			ri->getEdge(v1, v2, j);
			const Vector3F& p1 = pos[v1];
			glVertex3f(p1.x,p1.y,p1.z);
			const Vector3F& p2 = pos[v2];
			glVertex3f(p2.x,p2.y,p2.z);
		}
	}
	glEnd();
	
	glColor3f(1,1,0);
	glBegin(GL_LINE_STRIP);
	for(int i=0; i< nr;++i) {
		const pbd::ShapeMatchingRegion* ri = m_solver->region(i);
		const int& nv = ri->numPoints();
		for(int j=0;j<nv;++j) {
			glVertex3fv(ri->goalPosition(j) );
		}
	}
	glEnd();
	
	qDebug()<<" simulation step"<<m_solver->numTicks();

	drawWindTurbine();
	
	getDrawer()->m_markerProfile.apply();

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
		case Qt::Key_C:
			emit restartAtCurrentState();
			break;
		case Qt::Key_Space:
			emit pauseSim();
			//m_solver->simulate();
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
