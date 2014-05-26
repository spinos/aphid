/****************************************************************************
**
** Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial Usage
** Licensees holding valid Qt Commercial licenses may use this file in
** accordance with the Qt Commercial License Agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and Nokia.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights.  These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
** If you have questions regarding the use of this file, please contact
** Nokia at qt-info@nokia.com.
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtGui>
#include <QtOpenGL>
#include <DynamicsSolver.h>
#include "glwidget.h"

//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	m_state = new caterpillar::PhysicsState;
    m_vehicle = new caterpillar::TrackedPhysics;
	
	caterpillar::PhysicsState::engine->initPhysics();
	caterpillar::PhysicsState::engine->addGroundPlane(1000.f, -1.f);
	
	m_vehicle->setOrigin(Vector3F(0.f, 10.f, -10.f));
	m_vehicle->setSpan(82.5f);
	m_vehicle->setHeight(6.f);
	m_vehicle->setWidth(27.f);
	m_vehicle->setTrackWidth(7.8f);
	m_vehicle->setTensionerRadius(4.);
	m_vehicle->setNumRoadWheels(7);
	m_vehicle->setRoadWheelZ(0, 29.f);
	m_vehicle->setRoadWheelZ(1, 18.f);
	m_vehicle->setRoadWheelZ(2, 8.f);
	m_vehicle->setRoadWheelZ(3, -2.f);
	m_vehicle->setRoadWheelZ(4, -12.f);
	m_vehicle->setRoadWheelZ(5, -22.f);
	m_vehicle->setRoadWheelZ(6, -32.f);
	m_vehicle->setNumSupportRollers(3);
	m_vehicle->setSupportRollerZ(0, 24.f);
	m_vehicle->setSupportRollerZ(1, 2.f);
	m_vehicle->setSupportRollerZ(2, -20.f);
	m_vehicle->setTensionerY(.5f);
	m_vehicle->setDriveSprocketY(.5f);
	
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
//! [1]

caterpillar::TrackedPhysics* GLWidget::getSolver()
{
    return m_vehicle;
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
    update();
    caterpillar::PhysicsState::engine->simulate();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{	
	switch (e->key()) {
		case Qt::Key_T:
			m_vehicle->addTension(0.5f);
			break;
		case Qt::Key_W:
			m_vehicle->addPower(0.1f);
			break;
		case Qt::Key_S:
			m_vehicle->addPower(-0.1f);
			break;
		case Qt::Key_A:
			m_vehicle->addBrake(true);
			break;
		case Qt::Key_D:
			m_vehicle->addBrake(false);
			break;
		case Qt::Key_B:
			m_vehicle->addBrake(true);
			m_vehicle->addBrake(false);
			break;
		case Qt::Key_Space:
			const bool en = caterpillar::PhysicsState::engine->isPhysicsEnabled();
			caterpillar::PhysicsState::engine->setEnablePhysics(!en);
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(e);
}
