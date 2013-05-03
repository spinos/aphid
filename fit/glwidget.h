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

#include <QGLWidget>
#include <Base3DView.h>

class BaseMesh;
class KdTree;
class HarmonicCoord;
class AnchorGroup;
class Anchor;
class Ray;
class FitDeformer;

//! [0]
class GLWidget : public Base3DView
{
    Q_OBJECT

public:
    enum InteractMode {
        SelectCompnent,
        TransformAnchor
    };
    GLWidget(QWidget *parent = 0);
    ~GLWidget();

	virtual void clientDraw();
    virtual void clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir);
    virtual void sceneCenter(Vector3F & dst) const;
    
	void anchorSelected(float wei);
	void startDeform();
	bool pickupComponent(const Ray & ray, Vector3F & hit);
	
	void setTarget(AnchorGroup * src, KdTree * tree);
	void fit();
	void fitAnchors(Anchor * src, Anchor * dst);
	void removeLastAnchor();
	void buildTree();
	void loadMesh(std::string filename);
	void saveMesh(std::string filename);

protected:
    void keyPressEvent(QKeyEvent *event);
    
//! [3]
private:
    BaseMesh * m_mesh;
	KdTree * m_tree;
	InteractMode m_mode;
	AnchorGroup * m_anchors;
	FitDeformer * m_deformer;
	AnchorGroup * m_targetAnchors;
private slots:
    void simulate();
	
public slots:
	void open();
	void save();
};
//! [3]

#endif
