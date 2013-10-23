/*
 *  PoseEdit.cpp
 *  eulerRot
 *
 *  Created by jian zhang on 10/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "SkeletonPoseEdit.h"
#include <SkeletonSystem.h>
#include <ActionIconFrame.h>
#include <ToolContext.h>
#include "PoseListModel.h"

SkeletonPoseEdit::SkeletonPoseEdit(SkeletonSystem * skeleton, QWidget *parent) : QWidget(parent)
{
	m_skeleton = skeleton;
	
	ActionIconFrame * addPoseBtn = new ActionIconFrame(this);
	addPoseBtn->addIconFile(":addFile.png");
	addPoseBtn->addIconFile(":addFileActive.png");
	addPoseBtn->setIconIndex(0);
	addPoseBtn->setAction(ToolContext::AddSkeletonPose);
	
	ActionIconFrame * savePoseBtn = new ActionIconFrame(this);
	savePoseBtn->addIconFile(":save.png");
	savePoseBtn->addIconFile(":saveActive.png");
	savePoseBtn->setIconIndex(0);
	savePoseBtn->setAction(ToolContext::SaveSkeletonPose);
	
	model = new FileListModel(this);
    model->setSkeleton(m_skeleton);
	
	poseList = new QListView;
	poseList->setModel(model);
	
	QHBoxLayout * tools = new QHBoxLayout;
    tools->addWidget(addPoseBtn);
	tools->addWidget(savePoseBtn);
	tools->addStretch();
	
    QVBoxLayout * main = new QVBoxLayout;
	main->addLayout(tools);
    main->addWidget(poseList);
    setLayout(main);
	
	connect(addPoseBtn, SIGNAL(actionTriggered(int)), model, SLOT(addPose()));
	connect(savePoseBtn, SIGNAL(actionTriggered(int)), this, SLOT(savePose()));
	
	connect(poseList->selectionModel(),
            SIGNAL(selectionChanged(const QItemSelection &,
                                    const QItemSelection &)),
            this, SLOT(selectPose()));
}

SkeletonPoseEdit::~SkeletonPoseEdit() {}

void SkeletonPoseEdit::selectPose()
{
	if(poseList->selectionModel()->selection().isEmpty()) return;
	
	QModelIndex selectedIndex = poseList->selectionModel()->currentIndex();
	QVariant varName = model->data(selectedIndex, Qt::DisplayRole);
    m_skeleton->selectPose(varName.toString().toUtf8().data());
	m_skeleton->recoverPose();
	emit poseChanged();
}

void SkeletonPoseEdit::savePose()
{
	if(poseList->selectionModel()->selection().isEmpty()) return;
	m_skeleton->updatePose();
}