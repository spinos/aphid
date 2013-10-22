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

#include <math.h>

#include "glwidget.h"

#include "KdTreeDrawer.h"
#include <Anchor.h>
#include <KdTree.h>
#include <IntersectionContext.h>
#include <BaseTransform.h>
#include <SkeletonJoint.h>
#include <BaseBrush.h>
#include <TransformManipulator.h>
#include <SkeletonJoint.h>
#include <SkeletonSystem.h>

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    m_skeleton = new SkeletonSystem;
	SkeletonJoint * grp0 = new SkeletonJoint;
	grp0->setName("/humerus");
	grp0->setTranslation(Vector3F(12.f, 0.f, 0.f));
	grp0->setJointOrient(Vector3F(0.f, .3f, 0.f));
	m_skeleton->addJoint(grp0);
	
	SkeletonJoint * child1 = new SkeletonJoint(grp0);
	child1->setName("/humerus/ulna");
	child1->setTranslation(Vector3F(50.f, 0.f, 0.f));
	child1->setJointOrient(Vector3F(0.f, -.6f, 0.f));
	child1->setRotateDOF(Float3(0.f, 1.f, 0.f));
	grp0->addChild(child1);
	m_skeleton->addJoint(child1);
	
	SkeletonJoint * child2 = new SkeletonJoint(child1);
	child2->setName("/humerus/ulna/radius");
	child2->setTranslation(Vector3F(25.f, 0.f, 0.f));
	child2->setJointOrient(Vector3F(0.f, 0.f, 0.f));
	child2->setRotateDOF(Float3(1.f, 0.f, 0.f));
	child1->addChild(child2);
	m_skeleton->addJoint(child2);
	
	SkeletonJoint * child3 = new SkeletonJoint(child2);
	child3->setName("/humerus/ulna/radius/carpometacarpus");
	child3->setTranslation(Vector3F(25.f, 0.f, 0.f));
	child3->setJointOrient(Vector3F(0.f, .6f, 0.f));
	child3->setRotateDOF(Float3(0.f, 1.f, 1.f));
	child2->addChild(child3);
	m_skeleton->addJoint(child3);
	
	SkeletonJoint * child4 = new SkeletonJoint(child3);
	child4->setName("/humerus/ulna/radius/carpometacarpus/secondDigit");
	child4->setTranslation(Vector3F(30.f, 0.f, 0.f));
	child4->setJointOrient(Vector3F(0.f, 0.f, 0.f));
	child4->setRotateDOF(Float3(0.f, 1.f, 1.f));
	child3->addChild(child4);
	m_skeleton->addJoint(child4);
	
	SkeletonJoint * child5 = new SkeletonJoint(child4);
	child5->setName("/humerus/ulna/radius/carpometacarpus/secondDigit/digitEnd");
	child5->setTranslation(Vector3F(20.f, 0.f, 0.f));
	child5->setJointOrient(Vector3F(0.f, 0.f, 0.f));
	child5->setRotateDOF(Float3(0.f, 0.f, 0.f));
	child4->addChild(child5);
	m_skeleton->addJoint(child5);
	
	std::cout<<"skeleton dof "<<m_skeleton->degreeOfFreedom();
	
	solve();
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	getDrawer()->coordsys(m_space0, 8.f);
	getDrawer()->coordsys(m_space1, 8.f);

	for(unsigned i = 0; i < m_skeleton->numJoints(); i++) {
		getDrawer()->skeletonJoint(m_skeleton->joint(i));
	}
	
	getDrawer()->manipulator(manipulator());
}

void GLWidget::clientSelect()
{	
	const Ray * ray = getIncidentRay();

	SkeletonJoint * subject = m_skeleton->selectJoint(*ray);
	if(!subject) return;
	manipulator()->attachTo(subject);
	manipulator()->start(ray);
	emit jointSelected((int)subject->index());
}

void GLWidget::clientDeselect()
{
}

void GLWidget::clientMouseInput()
{
	const Ray * ray = getIncidentRay();
	manipulator()->perform(ray);
	emit jointChanged();
}

void GLWidget::setAngleAlpha(double x)
{
	m_angles.x = x;
	solve();
}

void GLWidget::setAngleBeta(double x)
{
	m_angles.y = x;
	solve();
}

void GLWidget::setAngleGamma(double x)
{
	m_angles.z = x;
	solve();
}

void GLWidget::solve()
{
	m_space0.setIdentity();
	m_space0.rotateEuler(DegreeToAngle(m_angles.x), DegreeToAngle(m_angles.y), DegreeToAngle(m_angles.z));
	m_space0.rotateEuler(0.2, 0.3, 0.5);
	
	m_space1.setIdentity();
	m_space1.rotateEuler(DegreeToAngle(m_angles.x), DegreeToAngle(m_angles.y), DegreeToAngle(m_angles.z));
	Matrix33F s; s.rotateEuler(0.2, 0.3, 0.5);
	m_space1.multiply(s);
	Vector3F dv(DegreeToAngle(m_angles.x), DegreeToAngle(m_angles.y), DegreeToAngle(m_angles.z));
	
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	if(e->key() == Qt::Key_A) {
		if(!manipulator()->isDetached()) {
			((SkeletonJoint *)(manipulator()->subject()))->align();
			manipulator()->reattach();
			emit jointChanged();
		}
	}
	Base3DView::keyPressEvent(e);
}

SkeletonSystem * GLWidget::skeleton() const
{
	return m_skeleton;
}

void GLWidget::updateJoint()
{
	if(!manipulator()->isDetached()) manipulator()->reattach();
	update();
}
