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
#include <RadialBasisFunction.h>

static Vector3F rayo(15.299140, 20.149620, 97.618355), raye(-141.333694, -64.416885, -886.411499);
	
//! [0]
GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
	QTimer *timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(30);
	
	m_drawer = new KdTreeDrawer;
	
	m_mode = SelectCompnent;
	
	RadialBasisFunction *rbf = new RadialBasisFunction;
	rbf->create(5);
	rbf->setXi(0, Vector3F(0,0,0));
	rbf->setXi(1, Vector3F(1,0.1,0));
	rbf->setXi(2, Vector3F(0.1,1.1,0));
	rbf->setXi(3, Vector3F(-1,0.2,0.1));
	rbf->setXi(4, Vector3F(-0.2,-2.2,-0.1));
	
	rbf->setCi(0, 1.0);
	rbf->setCi(1, 0.0);
	rbf->setCi(2, 0.0);
	rbf->setCi(3, 0.0);
	rbf->setCi(4, 0.0);
	rbf->setTau(2.0);
	rbf->computeWeights();
	
	float r = rbf->solve(Vector3F(0.0, 0.1, 0.1));
	qDebug()<<"rbf "<<r;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	if(m_mode != TransformAnchor) {
		m_drawer->setWired(1);
		m_drawer->setGrey(0.9f);
		
	}
    else {
		m_drawer->setWired(0);
		
	}	
	m_drawer->setGrey(0.5f);
	//m_drawer->drawKdTree(m_tree);
	//m_drawer->setWired(0);
	m_drawer->setColor(0.f, 1.f, 0.4f);
	
	glTranslatef(20,0,0);
	m_drawer->setWired(1);
	m_drawer->setColor(0.f, 1.f, .4f);
	
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit)
{
	rayo = origin;
	raye = origin + displacement;
	
	Ray ray(rayo, raye);
	if(m_mode == SelectCompnent) {
	}
	else {
		
	}
}
//! [9]

void GLWidget::clientDeselect()
{

}

//! [10]
void GLWidget::clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir)
{
	rayo = origin;
	raye = origin + displacement;
	Ray ray(rayo, raye);
	if(m_mode == SelectCompnent) {
		Vector3F hit;
		pickupComponent(ray, hit);
	}
	else {
	    //if(!m_activeAnchor) return;
		//m_activeAnchor->translate(stir);
		//m_harm->solve();
		//m_deformer->solve();
	}
}
//! [10]

void GLWidget::simulate()
{
    update();
}

void GLWidget::anchorSelected(float wei)
{
}

void GLWidget::startDeform()
{
}

bool GLWidget::pickupAnchor(const Ray & ray, Vector3F & hit)
{
	return  false;
}

bool GLWidget::pickupComponent(const Ray & ray, Vector3F & hit)
{
	return false;
}

