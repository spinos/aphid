/*
 *  PoseEdit.h
 *  eulerRot
 *
 *  Created by jian zhang on 10/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <QWidget>

class PoseSpaceDeformer;
class FileListModel;

QT_BEGIN_NAMESPACE
class QListView;
QT_END_NAMESPACE

class SkeletonPoseEdit : public QWidget {
Q_OBJECT
public:
    SkeletonPoseEdit(PoseSpaceDeformer * deformer, QWidget *parent = 0);
    virtual ~SkeletonPoseEdit();
	
protected:

private:
	QListView * poseList;
	FileListModel * model;

public slots:
    
private slots:
	void selectPose();
	void savePose();
	
signals:
	void poseChanged();
};
