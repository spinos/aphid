/*
 *  ManipulateView.h
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <deque>
#include <Base3DView.h>
#include <SelectionContext.h>

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
	
protected:
	bool pickupComponent(const Ray & ray, Vector3F & hit);
	bool hitTest();
	void selectFaces(SelectionContext::SelectMode m = SelectionContext::Replace);
	
	virtual Vector3F sceneCenter() const;
    
	virtual void buildTree();

	virtual void drawIntersection() const;
	
	void setRebuildTree();
	bool shouldRebuildTree() const;

	KdTree * getTree() const;
	virtual PatchMesh * activeMesh() const;
	
	TransformManipulator * manipulator();
	MeshManipulator * sculptor();
	
	const std::deque<unsigned> & selectedQue() const;
	
	virtual void clientDraw();
    virtual void clientSelect(QMouseEvent *event);
    virtual void clientDeselect(QMouseEvent *event);
    virtual void clientMouseInput(QMouseEvent *event);
    
    virtual void keyPressEvent(QKeyEvent *event);
    virtual void focusInEvent(QFocusEvent * event);
	virtual void clearSelection();
	
	void showManipulator() const;
    void showActiveFaces() const;
	
public slots:

private:
	bool isSelectingComponent() const;
	void selectComponent(QMouseEvent *event);
	bool isTransforming() const;
	void startTransform(QMouseEvent *event);
	void doTransform(QMouseEvent *event);
	void endTransform();
private:
	TransformManipulator * m_manipulator;
	MeshManipulator * m_sculptor;
	KdTree * m_tree;
	SelectionContext * m_selectCtx;
	bool m_shouldRebuildTree;
};
