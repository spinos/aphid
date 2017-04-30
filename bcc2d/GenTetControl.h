/*
 *  GenTetControl.h
 *  cudafem
 *
 *  Created by jian zhang on Mon Jul 27 18:34:17 CST 2015 
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef PHYSICS_CONTROL_H
#define PHYSICS_CONTROL_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QGroupBox;
class QCheckBox;
class QLabel;
class QPushButton;
class QComboBox;
QT_END_NAMESPACE

class QIntEditSlider;
class QDoubleEditSlider;
class QSplineEdit;
class QPolarCoordinateEdit;

class GenTetControl : public QDialog
{
    Q_OBJECT

public:
    GenTetControl(QWidget *parent = 0);
	
public slots:
	void receiveEstimatedN(unsigned x);
    
private slots:
	void sendRebuild();
    void sendPatchMethod(int x);
signals:
	void rebuildTet(double n);
    void patchMethodChanged(int x);
private:
    QDoubleEditSlider * m_estimateNValue;
    QPushButton * m_rebuildAct;
    QComboBox * m_patchMethodChooser;
};
#endif