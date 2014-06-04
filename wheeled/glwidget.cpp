#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <DynamicsSolver.h>
#include "glwidget.h"
#include <Obstacle.h>
//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_state = new caterpillar::PhysicsState;
    m_vehicle = new caterpillar::WheeledVehicle;
	
	caterpillar::PhysicsState::engine->initPhysics();
	caterpillar::PhysicsState::engine->addGroundPlane(2000.f, 0.f);
	
	caterpillar::Obstacle obst;
	obst.create(2000.f);
	
	m_vehicle->setOrigin(Vector3F(0.f, 17.f, 0.f));
	
	caterpillar::Wheel::Profile rearWheelInfo;
	rearWheelInfo._width = 2.89f;
	m_vehicle->setWheelInfo(1, rearWheelInfo);
	
	caterpillar::Suspension::Profile rearBridgeInfo;
	rearBridgeInfo._steerable = false;
	rearBridgeInfo._powered = true;
	m_vehicle->setSuspensionInfo(1, rearBridgeInfo);
	
	m_vehicle->create();
	std::cout<<"object groups "<<m_vehicle->str();
	
	caterpillar::PhysicsState::engine->setEnablePhysics(false);
	caterpillar::PhysicsState::engine->setNumSubSteps(12);
	caterpillar::PhysicsState::engine->setSimulateFrequency(180.f);
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
	m_vehicle->displayStatistics();
    update();
	m_vehicle->update();
    caterpillar::PhysicsState::engine->simulate();
	getCamera()->traverse(m_vehicle->vehicleTraverse());
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{	
    bool enabled;
	switch (e->key()) {
		case Qt::Key_C:
			m_vehicle->setSteerAngle(0.f);
			break;
		case Qt::Key_W:
			m_vehicle->addTargetSpeed(2.99f);
			break;
		case Qt::Key_S:
			m_vehicle->addTargetSpeed(-1.31f);
			break;
		case Qt::Key_A:
			m_vehicle->addSteerAngle(-0.01f);
			break;
		case Qt::Key_D:
			m_vehicle->addSteerAngle(0.01f);
			break;
		case Qt::Key_B:
			m_vehicle->setTargetSpeed(0.f);
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
