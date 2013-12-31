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

class MlUVView : public Base2DView, public FeatherExample {
Q_OBJECT

public:
    MlUVView(QWidget *parent = 0);
    ~MlUVView();
	
	virtual void clientDraw();
	virtual void clientSelect();
	virtual void clientMouseInput();
	
	void addFeather();
	void removeSelectedFeather();
    void changeSelectedFeatherNSegment(int d);
	void chooseImageBackground(std::string & name);
	void loadImageBackground(const std::string & name);
signals:
	void selectionChanged();
	void shapeChanged();
private:
	bool pickupFeather(const Vector2F & p);
	void drawFeather(MlFeather * f);
	void drawControlVectors(MlFeather * f);
	void drawActiveBound();
private:
	float * m_selectedVert;
	Vector2F m_selectVertWP;
	int m_texId;
	bool m_moveYOnly;
};