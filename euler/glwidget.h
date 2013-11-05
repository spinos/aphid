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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <Base3DView.h>
class SkeletonJoint;
class SkeletonSystem;
class PlaneMesh;
class PoseSpaceDeformer;

class GLWidget : public Base3DView
{
    Q_OBJECT

public:
    GLWidget(QWidget *parent = 0);
    virtual ~GLWidget();
	
	SkeletonSystem * skeleton() const;

protected:
    virtual void clientDraw();
    virtual void clientSelect();
	virtual void clientDeselect();
	virtual void clientMouseInput();
	virtual void keyPressEvent(QKeyEvent *event);
private:
	void solve();
	void selectJoint();
	void changeJoint();
	void changeMesh();
	void selectVertex();
private:
	Matrix33F m_space0, m_space1;
	Vector3F m_angles;
	
	SkeletonSystem * m_skeleton;
	PlaneMesh * m_mesh;
	PoseSpaceDeformer * m_deformer;
	
private slots:
	
public slots:
	void setAngleAlpha(double x);
	void setAngleBeta(double x);
	void setAngleGamma(double x);
	void updateJoint();
	
signals:
    void jointSelected(int idx);
	void jointChanged();
};
//! [3]

#endif
