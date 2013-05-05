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

class BaseMesh;
class KdTree;
class AnchorGroup;
class Anchor;
class Ray;

class SingleModelView : public Base3DView
{
    Q_OBJECT

public:
    enum InteractMode {
        SelectCompnent,
        TransformAnchor
    };
	
    SingleModelView(QWidget *parent = 0);
    ~SingleModelView();
	
	void anchorSelected(float wei);
	bool pickupComponent(const Ray & ray, Vector3F & hit);
	
	virtual void clientDraw();
    virtual void clientSelect(Vector3F & origin, Vector3F & displacement, Vector3F & hit);
    virtual void clientDeselect();
    virtual void clientMouseInput(Vector3F & origin, Vector3F & displacement, Vector3F & stir);
    virtual void sceneCenter(Vector3F & dst) const;
    
	virtual void buildTree();
	virtual void loadMesh(std::string filename);
	virtual void saveMesh(std::string filename);
	
	void setSelectComponent();
	void setSelectAnchor();
	
	void drawAnchors();
	AnchorGroup * getAnchors() const;
	KdTree * getTree() const;
	
protected:
    void keyPressEvent(QKeyEvent *event);
    
public:
    BaseMesh * m_mesh;
	KdTree * m_tree;
	InteractMode m_mode;
	AnchorGroup * m_anchors;
	
private:
	
public slots:
	void open();
	void save();
};
