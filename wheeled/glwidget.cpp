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
	//åcaterpillar::PhysicsState::engine->addGroundPlane(2000.f, 0.f);
	
	//caterpillar::Obstacle obst;
	//obst.create(2000.f);
	
	m_circuit = new caterpillar::RaceCircuit;
	m_circuit->create();
	
	m_vehicle->setOrigin(Vector3F(0.f, 20.f, -10.f));
	getCamera()->traverse(Vector3F(0.f, 20.f, -10.f));
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	
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
	caterpillar::PhysicsState::engine->setNumSubSteps(18);
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
	int i = 1;
	std::stringstream sst;
	sst.str("");
	sst<<"vehicle speed: "<<m_vehicle->vehicleVelocity().length();
	hudText(sst.str(), i++);
	sst.str("");
	sst<<"vehicle acceleration: "<<m_vehicle->acceleration();
	hudText(sst.str(), i++);
	sst.str("");
	sst<<"gear: "<<m_vehicle->gear();
	hudText(sst.str(), i++);
	sst.str("");
	sst<<"vehicle drifting: "<<m_vehicle->drifting();
	hudText(sst.str(), i++);
	sst.str("");
	sst<<"gas strength: "<<m_vehicle->gasStrength();
	hudText(sst.str(), i++);
	sst.str("");
	sst<<"bake strength: "<<m_vehicle->brakeStrength();
	hudText(sst.str(), i++);
	sst.str("");
	sst<<"turn angle: "<<m_vehicle->turnAngle();
	hudText(sst.str(), i++);
	sst.str("");
	float t[2];
	m_vehicle->wheelForce(0, t);
	sst<<"force front wheel: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelForce(1, t);
	sst.str("");
	sst<<"force back wheel: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelSlip(0, t);
	sst.str("");
	sst<<"slip front wheel: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelSlip(1, t);
	sst.str("");
	sst<<"slip back wheel: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelSkid(0, t);
	sst.str("");
	sst<<"skid front wheel: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelSkid(1, t);
	sst.str("");
	sst<<"skid back wheel: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
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
	update();
	m_vehicle->update();
    caterpillar::PhysicsState::engine->simulate();
    getCamera()->traverse(m_vehicle->vehicleTraverse());
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{	
    bool enabled;
	switch (e->key()) {
		case Qt::Key_W:
		    m_vehicle->addGas(.13f);
			break;
		case Qt::Key_B:
			m_vehicle->addBrakeStrength(.23f);
			break;
		case Qt::Key_P:
			m_vehicle->setParkingBrake(true);
			break;
		case Qt::Key_A:
			m_vehicle->addSteerAngle(0.009f);
			break;
		case Qt::Key_S:
			m_vehicle->setSteerAngle(0.f);
			break;
		case Qt::Key_D:
			m_vehicle->addSteerAngle(-0.009f);
			break;
		case Qt::Key_F:
			m_vehicle->changeGear(1);
			break;
		case Qt::Key_C:
			m_vehicle->changeGear(-1);
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
		    if(event->isAutoRepeat()) event->ignore();
			else m_vehicle->setGas(0.f);
			break;
		case Qt::Key_B:
			if(event->isAutoRepeat()) event->ignore();
			else m_vehicle->setBrakeStrength(0.f);
			break;
		case Qt::Key_P:
			if(event->isAutoRepeat()) event->ignore();
			else m_vehicle->setParkingBrake(false);
			break;
		default:
			break;
	}
	
	Base3DView::keyReleaseEvent(event);
}

