/*
 *  SingleModelView.h
 *  fit
 *
 *  Created by jian zhang on 5/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <Base3DView.h>

class ToolContext;
class PatchMesh;
class KdTree;
class AnchorGroup;
class Anchor;
class Ray;
class MeshTopology;

class SingleModelView : public Base3DView
{
    Q_OBJECT

public:
    SingleModelView(QWidget *parent = 0);
    ~SingleModelView();
	
	virtual bool anchorSelected(float wei);
	virtual bool removeLastAnchor(unsigned & idx);
	virtual bool removeActiveAnchor(unsigned & idx);
	
	bool pickupComponent(const Ray & ray, Vector3F & hit);
	bool hitTest(const Ray & ray, Vector3F & hit);
	
	virtual void clientDraw();
    virtual void clientSelect();
    virtual void clientDeselect();
    virtual void clientMouseInput();
    virtual Vector3F sceneCenter() const;
    
	virtual void buildTree();
	virtual void buildTopology();
	virtual void loadMesh(std::string filename);
	virtual void saveMesh(std::string filename);
	
	virtual void drawIntersection() const;
	
	void clearTree();
	void clearTopology();
	void setRebuildTree();
	bool shouldRebuildTree() const;

	void drawAnchors();
	AnchorGroup * getAnchors() const;
	KdTree * getTree() const;
	virtual PatchMesh * mesh();
	
protected:
    virtual void keyPressEvent(QKeyEvent *event);
    virtual void focusInEvent(QFocusEvent * event);
	
public:
	KdTree * m_tree;
	AnchorGroup * m_anchors;
	MeshTopology * m_topo;
	
public slots:
	void open();
	void save();
	
private:
	PatchMesh * m_mesh;
	bool m_shouldRebuildTree;
};
