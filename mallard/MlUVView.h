/*
 *  MlUVView.h
 *  mallard
 *
 *  Created by jian zhang on 10/2/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Base2DView.h>
#include <FeatherExample.h>
class MlFeather;
class BaseVane;
class MlUVView : public Base2DView, public FeatherExample {
Q_OBJECT

public:
    MlUVView(QWidget *parent = 0);
    ~MlUVView();
	
	virtual void clientDraw();
	virtual void clientSelect(QMouseEvent *event);
	virtual void clientMouseInput(QMouseEvent *event);
	
	void addFeather();
	void removeSelectedFeather();
    void changeSelectedFeatherNSegment(int d);
	void chooseImageBackground(std::string & name);
	void loadImageBackground(const std::string & name);
	void changeSelectedFeatherType();
signals:
	void selectionChanged();
	void shapeChanged();
private:
	bool pickupFeather(const Vector2F & p);
	void drawFeather(MlFeather * f);
	void drawControlVectors(MlFeather * f);
	void drawVaneVectors(BaseVane * vane);
	void drawBindVectors(MlFeather * f);
	void drawActiveBound();
private:
	float * m_selectedVert;
	Vector2F m_selectVertWP;
	int m_texId;
	bool m_moveYOnly;
};