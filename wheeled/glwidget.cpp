#include <QtGui>
#include <QtOpenGL>
#include <DynamicsSolver.h>
#include "glwidget.h"

//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_state = new caterpillar::PhysicsState;
    m_vehicle = new caterpillar::WheeledVehicle;
	
	caterpillar::PhysicsState::engine->initPhysics();
	caterpillar::PhysicsState::engine->addGroundPlane(1000.f, -1.f);
	
	m_vehicle->setOrigin(Vector3F(-12.f, 17.f, -20.f));
	
	caterpillar::Wheel::Profile rearWheelInfo;
	rearWheelInfo._width = 2.89f;
	m_vehicle->setWheelInfo(1, rearWheelInfo);
	
	m_vehicle->create();
	std::cout<<"object groups "<<m_vehicle->str();
	
	caterpillar::PhysicsState::engine->setEnablePhysics(false);
	caterpillar::PhysicsState::engine->setNumSubSteps(10);
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(30);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
	delete m_vehicle;
	delete m_state;
}

//! [7]
void GLWidget::clientDraw()
{
	caterpillar::PhysicsState::engine->renderWorld();
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void GLWidget::clientDeselect()
{
}

//! [10]
void GLWidget::clientMouseInput(Vector3F & stir)
{
}
//! [10]

void GLWidget::simulate()
{
	//m_vehicle->displayStatistics();
    update();
    caterpillar::PhysicsState::engine->simulate();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{	
    bool enabled;
	switch (e->key()) {
		case Qt::Key_T:
			//m_vehicle->addTension(100.f);
			break;
		case Qt::Key_R:
			//m_vehicle->addTension(-100.f);
			break;
		case Qt::Key_W:
			//m_vehicle->addPower(0.1f);
			break;
		case Qt::Key_S:
			//m_vehicle->addPower(-0.1f);
			break;
		case Qt::Key_A:
			//m_vehicle->addBrake(true);
			break;
		case Qt::Key_D:
			//m_vehicle->addBrake(false);
			break;
		case Qt::Key_B:
			//m_vehicle->addBrake(true);
			//m_vehicle->addBrake(false);
			break;
		case Qt::Key_Space:
			enabled = caterpillar::PhysicsState::engine->isPhysicsEnabled();
			caterpillar::PhysicsState::engine->setEnablePhysics(!enabled);
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(e);
}
