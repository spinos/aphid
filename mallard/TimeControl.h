/*
 *  TimeControl.h
 *  mallard
 *
 *  Created by jian zhang on 10/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once
#include <PlaybackControl.h>
#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QScrollBar;
class QSpinBox;
class QLabel;
QT_END_NAMESPACE

class QIntEditSlider;

class TimeControl : public QDialog, public PlaybackControl
{
    Q_OBJECT

public:
    TimeControl(QWidget *parent = 0);
	
	virtual void setFrameRange(int mn, int mx);
	virtual void disable();
	virtual void enable();
	virtual int playbackMin() const;
	virtual int playbackMax() const;
	
public slots:
	void updateCurrentFrame(int x);
	
private slots:
	void setPlayMin();
	void setPlayMax();
	
signals:
	void currentFrameChanged(int a);
	
private:
	QGroupBox *minGroup;
	QGroupBox *playGroup;
	QGroupBox *maxGroup;
	QScrollBar * bar;
	QLabel * maxLabel;
	QLabel * minLabel;
	QSpinBox * minSpin;
	QSpinBox * maxSpin;
	QSpinBox * currentSpin;
};
