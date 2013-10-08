/*
 *  TimeControl.h
 *  mallard
 *
 *  Created by jian zhang on 10/8/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
 
#pragma once

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QScrollBar;
class QSpinBox;
class QLabel;
QT_END_NAMESPACE

class TimeControl : public QDialog
{
    Q_OBJECT

public:
    TimeControl(QWidget *parent = 0);
	
private slots:
	
signals:
	
private:
	QGroupBox *controlsGroup;
	QScrollBar * m_bar;
	QLabel * m_maxLabel;
	QLabel * m_minLabel;
	QLabel * m_currentLabel;
	QSpinBox * m_minSpin;
	QSpinBox * m_maxSpin;
	QSpinBox * m_currentSpin;
};
