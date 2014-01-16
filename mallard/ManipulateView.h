/*
 *  ManipulateView.h
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <Base3DView.h>

class PatchMesh;
class KdTree;
class TransformManipulator;
class MeshManipulator;

class ManipulateView : public Base3DView
{
    Q_OBJECT

public:
    ManipulateView(QWidget *parent = 0);
    ~ManipulateView();
	
	bool pickupComponent(const Ray & ray, Vector3F & hit);
	bool hitTest(const Ray & ray, Vector3F & hit);
	
	virtual void clientDraw();
    virtual void clientSelect();
    virtual void clientDeselect();
    virtual void clientMouseInput();
    virtual Vector3F sceneCenter() const;
    
	virtual void buildTree();

	virtual void drawIntersection() const;
	
	void setRebuildTree();
	bool shouldRebuildTree() const;

	KdTree * getTree() const;
	virtual PatchMesh * activeMesh();
	
	TransformManipulator * manipulator();
	MeshManipulator * sculptor();
	
protected:
    virtual void keyPressEvent(QKeyEvent *event);
    virtual void focusInEvent(QFocusEvent * event);
	virtual void clearSelection();
	virtual void processSelection(QMouseEvent *event);
    virtual void processDeselection(QMouseEvent *event);
	void showManipulator() const;
    
public slots:
	
private:
	TransformManipulator * m_manipulator;
	MeshManipulator * m_sculptor;
	KdTree * m_tree;
	bool m_shouldRebuildTree;
};
