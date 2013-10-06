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
class MlFeatherCollection;
class MlFeather;
class ZEXRImage;
class MlUVView : public Base2DView {
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
	void loadImageBackground();
	
	static MlFeatherCollection * FeatherLibrary;

private:
	bool pickupFeather(const Vector2F & p);
	void drawFeather(MlFeather * f);
	void drawControlVectors(MlFeather * f);
	void drawActiveBound();
private:
	float * m_selectedVert;
	ZEXRImage* m_image;
	Vector2F m_selectVertWP;
	int m_activeId;
	int m_texId;
	bool m_moveYOnly;
};