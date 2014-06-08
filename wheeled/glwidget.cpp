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
	
	m_vehicle->setOrigin(Vector3F(7.f, 17.f, -1900.f));
	getCamera()->traverse(Vector3F(7.f, 17.f, -1900.f));
	
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
	caterpillar::PhysicsState::engine->setNumSubSteps(15);
	caterpillar::PhysicsState::engine->setEnableDrawConstraint(false);
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

	std::stringstream sst;
	sst.str("");
	sst<<"vehicle speed: "<<m_vehicle->vehicleVelocity().length();
	hudText(sst.str(), 1);
	sst.str("");
	sst<<"gas strength: "<<m_vehicle->gasStrength();
	hudText(sst.str(), 2);
	sst.str("");
	sst<<"bake strength: "<<m_vehicle->brakeStrength();
	hudText(sst.str(), 3);
	sst.str("");
	sst<<"turn angle: "<<m_vehicle->turnAngle();
	hudText(sst.str(), 4);
	sst.str("");
	float diff[2];
	m_vehicle->differential(0, diff);
	sst<<"differential[0]: "<<diff[0]<<","<<diff[1];
	hudText(sst.str(), 5);
	m_vehicle->differential(1, diff);
	sst.str("");
	sst<<"differential[1]: "<<diff[0]<<","<<diff[1];
	hudText(sst.str(), 6);
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
	getCamera()->traverse(m_vehicle->vehicleTraverse());
    update();
	m_vehicle->update();
    caterpillar::PhysicsState::engine->simulate();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{	
    bool enabled;
	switch (e->key()) {
		case Qt::Key_C:
			m_vehicle->setSteerAngle(0.f);
			break;
		case Qt::Key_W:
			m_vehicle->addGas(.13f);
			break;
		case Qt::Key_B:
			m_vehicle->addBrakeStrength(.23f);
			break;
		case Qt::Key_S:
			m_vehicle->setGoForward(!m_vehicle->goingForward());
			break;
		case Qt::Key_A:
			m_vehicle->addSteerAngle(0.019f);
			break;
		case Qt::Key_D:
			m_vehicle->addSteerAngle(-0.019f);
			break;
		case Qt::Key_F:
			m_vehicle->setBrakeStrength(1.f);
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

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_W:
			m_vehicle->setGas(0.f);
			break;
		case Qt::Key_B:
			m_vehicle->setBrakeStrength(0.f);
			break;
		case Qt::Key_F:
			m_vehicle->setBrakeStrength(0.f);
			break;
		default:
			break;
	}
	
	Base3DView::keyReleaseEvent(event);
}

