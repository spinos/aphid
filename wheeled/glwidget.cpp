#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include <DynamicsSolver.h>
#include "glwidget.h"
#include <Obstacle.h>
#include <KdTreeDrawer.h>

//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_state = new caterpillar::PhysicsState;
    m_vehicle = new caterpillar::Automobile;
	
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

#ifdef PORSCHE	
	caterpillar::Wheel::Profile rearWheelInfo;
	rearWheelInfo._width = 2.9f;
	m_vehicle->setWheelInfo(1, rearWheelInfo);
	caterpillar::Suspension::Profile rearBridgeInfo;
	rearBridgeInfo._steerable = false;
	rearBridgeInfo._powered = true;
	m_vehicle->setSuspensionInfo(1, rearBridgeInfo);
#else
    caterpillar::Wheel::Profile frontWheelInfo;
    frontWheelInfo._radiusMajor = 3.25f;
    frontWheelInfo._radiusMinor = .6f;
    frontWheelInfo._width = 2.9f;
	m_vehicle->setWheelInfo(0, frontWheelInfo);
    caterpillar::Wheel::Profile rearWheelInfo;
    rearWheelInfo._radiusMajor = 3.38f;
    rearWheelInfo._radiusMinor = .55f;
    rearWheelInfo._width = 2.9f;
	m_vehicle->setWheelInfo(1, rearWheelInfo);
	
	caterpillar::Suspension::Profile susp;
	caterpillar::Suspension::RodRadius = .3f;
	susp._damperY = 2.f;
	susp._upperJointY = 1.5f;
	susp._lowerJointY = -1.37f;
	susp._steerArmJointZ = 1.43f;
	susp._upperWishboneLength = 4.4f;
	susp._lowerWishboneLength = 5.7f;
	m_vehicle->setSuspensionInfo(0, susp);
	susp._steerable = false;
	susp._powered = true;
	m_vehicle->setSuspensionInfo(1, susp);
	
	m_vehicle->setHullDim(Vector3F(20.f, 4.f, 40.f));
	m_vehicle->setAxisCoord(0, 17.f, -.25f, 8.12f);
	m_vehicle->setAxisCoord(1, 17.f, -.25f, -10.63f);
	
#endif
	
	m_vehicle->create();
	std::cout<<"object groups "<<m_vehicle->str();
	
	caterpillar::PhysicsState::engine->setEnablePhysics(false);
	caterpillar::PhysicsState::engine->setNumSubSteps(18);
	caterpillar::PhysicsState::engine->setEnableDrawConstraint(false);
	caterpillar::PhysicsState::engine->setSimulateFrequency(200.f);
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
	getDrawer()->m_paintProfile.apply();
	m_vehicle->render();
	int i = 1;
	std::stringstream sst;
	sst.str("");
	sst<<"vehicle speed: "<<m_vehicle->velocity().length();
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
	sst<<"front wheel slip: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelSlip(1, t);
	sst.str("");
	sst<<"back wheel slip: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelSkid(0, t);
	sst.str("");
	sst<<"front wheel skid: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelSkid(1, t);
	sst.str("");
	sst<<"back wheel skid: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelFriction(0, t);
	sst.str("");
	sst<<"front wheel friction: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	m_vehicle->wheelFriction(1, t);
	sst.str("");
	sst<<"back wheel friction: "<<t[0]<<","<<t[1];
	hudText(sst.str(), i++);
	sst.str("");
	sst<<"downward force: "<<m_vehicle->downForce();
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
			m_vehicle->addSteerAngle(0.013f);
			break;
		case Qt::Key_S:
			m_vehicle->setSteerAngle(0.f);
			break;
		case Qt::Key_D:
			m_vehicle->addSteerAngle(-0.013f);
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

