/*
 *  BrushControl.h
 *  mallard
 *
 *  Created by jian zhang on 9/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef BRUSH_CONTROL_H
#define BRUSH_CONTROL_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QStackedLayout;
class QCheckBox;
QT_END_NAMESPACE

class QIntEditSlider;
class QDoubleEditSlider;
class SelectFaceBox;
class CombBox;
class CurlBox;
class ScaleBox;
class FloodBox;
class EraseBox;
class PaintBox;
class BaseBrush;

class BrushControl : public QDialog
{
    Q_OBJECT

public:
    BrushControl(BaseBrush * brush, QWidget *parent = 0);
	
public slots:
	void receiveToolContext(int c);
	
private slots:
	void sendBrushRadius(double d);
	void sendBrushStrength(double d);
	void sendBrushTwoSided(int x);
	void sendBrushFilterByColor(int x);
	void sendBrushPitch(double d);
	void sendBrushNumSamples(int x);
	void sendBrushColor(QColor c);
	void sendBrushDropoff(double x);
	void sendPaintMode(int x);
signals:
	void brushChanged();
	void paintModeChanged(int x);
	
private:
	QStackedLayout * stackLayout;
	SelectFaceBox * selectFace;
	CombBox * comb;
	CurlBox * curl;
	ScaleBox * brushScale;
	FloodBox * flood;
	EraseBox * eraseControl;
	PaintBox * paintControl;
	BaseBrush * m_brush;
};
#endif