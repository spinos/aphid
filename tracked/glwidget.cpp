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
    _dynamics = new caterpillar::TrackedPhysics;
	
	caterpillar::PhysicsState::engine->initPhysics();
	caterpillar::PhysicsState::engine->addGroundPlane(1000.f, -1.f);
	
	_dynamics->setOrigin(Vector3F(0.f, 10.f, -10.f));
	_dynamics->setSpan(81.f);
	_dynamics->setHeight(6.f);
	_dynamics->setWidth(27.f);
	_dynamics->setTensionerRadius(3.2);
	_dynamics->setNumRoadWheels(7);
	_dynamics->setRoadWheelZ(0, 29.f);
	_dynamics->setRoadWheelZ(1, 18.f);
	_dynamics->setRoadWheelZ(2, 8.f);
	_dynamics->setRoadWheelZ(3, -2.f);
	_dynamics->setRoadWheelZ(4, -12.f);
	_dynamics->setRoadWheelZ(5, -22.f);
	_dynamics->setRoadWheelZ(6, -32.f);
	_dynamics->setNumSupportRollers(3);
	_dynamics->setSupportRollerZ(0, 24.f);
	_dynamics->setSupportRollerZ(1, 2.f);
	_dynamics->setSupportRollerZ(2, -18.f);
	
	_dynamics->create();
	
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(30);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
	delete _dynamics;
	delete m_state;
}
//! [1]

caterpillar::TrackedPhysics* GLWidget::getSolver()
{
    return _dynamics;
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
			_dynamics->addTension(0.5f);
			break;
		case Qt::Key_W:
			_dynamics->addPower(0.1f);
			break;
		case Qt::Key_S:
			_dynamics->addPower(-0.1f);
			break;
		case Qt::Key_A:
			_dynamics->addBrake(true);
			break;
		case Qt::Key_D:
			_dynamics->addBrake(false);
			break;
		case Qt::Key_B:
			_dynamics->addBrake(true);
			_dynamics->addBrake(false);
			break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(e);
}
