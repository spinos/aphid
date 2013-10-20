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

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	SkeletonJoint * grp0 = new SkeletonJoint;
	grp0->setName("/group0");
	m_groups.push_back(grp0);
	
	SkeletonJoint * child1 = new SkeletonJoint(grp0);
	child1->setName("/group0/child1");
	child1->setTranslation(Vector3F(44.f, 0.f, 0.f));
	grp0->addChild(child1);
	m_groups.push_back(child1);
	
	SkeletonJoint * child2 = new SkeletonJoint(child1);
	child2->setName("/group0/child1/child2");
	child2->setTranslation(Vector3F(0.f, 0.f, 32.f));
	child1->addChild(child2);
	m_groups.push_back(child2);
	
	solve();
}

GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	getDrawer()->coordsys(m_space0, 8.f);
	getDrawer()->coordsys(m_space1, 8.f);

	std::vector<SkeletonJoint *>::iterator it = m_groups.begin();
	for(; it != m_groups.end(); ++it) {
		getDrawer()->skeletonJoint(*it);
	}
	
	getDrawer()->manipulator(manipulator());
}

void GLWidget::clientSelect()
{	
	const Ray * ray = getIncidentRay();
	std::vector<SkeletonJoint *>::iterator it = m_groups.begin();
	BaseTransform * subject = *it;
	for(; it != m_groups.end(); ++it) {
		if((*it)->intersect(*ray)) subject = *it;
	}
	manipulator()->attachTo(subject);
	manipulator()->start(ray);
}

void GLWidget::clientDeselect()
{
}

void GLWidget::clientMouseInput()
{
	const Ray * ray = getIncidentRay();
	manipulator()->perform(ray);
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
	((SkeletonJoint *)m_groups[0])->setJointOrient(dv);
	update();
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	if(e->key() == Qt::Key_A) {
		std::vector<SkeletonJoint *>::iterator it = m_groups.begin();
		for(; it != m_groups.end(); ++it) {
			(*it)->setRotationAngles(Vector3F(0.f, 0.f, 0.f));
			(*it)->align();
		}
			
		manipulator()->reattach();
	}
	Base3DView::keyPressEvent(e);
}
