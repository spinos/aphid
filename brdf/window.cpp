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

#include "glwidget.h"
#include "window.h"
#include "Lambert.h"
#include "Phong.h"
#include "Ward.h"
#include "Cooktorrance.h"

//! [0]
Window::Window()
{
    centralWidget = new QWidget;
    setCentralWidget(centralWidget);
    glWidget = new GLWidget;
    lambert = new Lambert;
    phong = new Phong;
    ward = new Ward;
	cooktorrance = new Cooktorrance;
    
    brdfCombo = new QComboBox;
    brdfCombo->addItem(tr("Lambert"));
    brdfCombo->addItem(tr("Phong"));
    brdfCombo->addItem(tr("Ward"));
	brdfCombo->addItem(tr("Cooktorrance"));
    
    controlStack = new QStackedWidget;
    controlStack->addWidget(lambert);
    controlStack->addWidget(phong);
    controlStack->addWidget(ward);
	controlStack->addWidget(cooktorrance);
    
    thetaName = new QLabel(tr("Theta of V"));
    thetaValue = new QLineEdit;
    thetaValue->setReadOnly(true);
    thetaValue->setText(tr("45"));
    thetaControl = new QSlider(Qt::Horizontal);
    thetaControl->setRange(1, 90);
	thetaControl->setSingleStep(1);
	thetaControl->setValue(45);
    
    QGridLayout * thetaLayout = new QGridLayout;
    thetaLayout->setColumnStretch(2, 1);
    thetaLayout->addWidget(thetaName, 0, 0);
    thetaLayout->addWidget(thetaValue, 0, 1);
    thetaLayout->addWidget(thetaControl, 0, 2);
    
	QVBoxLayout *mainLayout = new QVBoxLayout;
	mainLayout->addWidget(glWidget);
	mainLayout->addLayout(thetaLayout);
	mainLayout->addWidget(brdfCombo);
	mainLayout->addWidget(controlStack);
	centralWidget->setLayout(mainLayout);
	
    setWindowTitle(tr("CUDA BRDF Visualization"));
    
    glWidget->setProgram(lambert);
    
    connect(brdfCombo, SIGNAL(activated(int)),
            this, SLOT(chooseProgram(int)));
    
    connect(thetaControl, SIGNAL(valueChanged(int)),
            this, SLOT(setThetaOfV(int)));
}
//! [1]

//! [2]
// QSlider *Window::createSlider()
// {
    // QSlider *slider = new QSlider(Qt::Vertical);
    // slider->setRange(0, 360 * 16);
    // slider->setSingleStep(16);
    // slider->setPageStep(15 * 16);
    // slider->setTickInterval(15 * 16);
    // slider->setTickPosition(QSlider::TicksRight);
    // return slider;
// }
//! [2]

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

void Window::chooseProgram(int value)
{
    controlStack->setCurrentIndex(value);
    if(value == 0)
        glWidget->setProgram(lambert);
    else if(value == 1)
        glWidget->setProgram(phong);
    else if(value == 2)
        glWidget->setProgram(ward);
	else
		glWidget->setProgram(cooktorrance);
}

void Window::setThetaOfV(int value)
{
    float theta = (float)value / 90.f * 3.1415927f * 0.5f;
    QString t;
	t.setNum(value);
	thetaValue->setText(t);
    BRDFProgram::setVTheta(theta);
}
