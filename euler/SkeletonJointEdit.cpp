#include <QtGui>
#include "SkeletonJointEdit.h"
#include <SkeletonJoint.h>
#include <SkeletonSystem.h>
#include <QDouble3Edit.h>
#include <sstream>

SkeletonJointEdit::SkeletonJointEdit(SkeletonSystem * skeleton, QWidget *parent) : QWidget(parent)
{
	m_skeleton = skeleton;
    controlGrp = new QGroupBox(tr("unknown"));
	translation = new QDouble3Edit(tr("translate"));
	orientAngle = new QDouble3Edit(tr("orient"));
	rotateAngle = new QDouble3Edit(tr("rotate"));
	
	QVBoxLayout * controlLayout = new QVBoxLayout;
	controlLayout->addWidget(translation);
	controlLayout->addWidget(rotateAngle);
	controlLayout->addWidget(orientAngle);
	controlGrp->setLayout(controlLayout);
    
    QVBoxLayout * main = new QVBoxLayout;
    main->addWidget(controlGrp);
    setLayout(main);
	
	connect(translation, SIGNAL(valueChanged(Vector3F)),
            this, SLOT(setJointTranslation(Vector3F)));
			
	connect(rotateAngle, SIGNAL(valueChanged(Vector3F)),
            this, SLOT(setJointRotation(Vector3F)));
			
	connect(orientAngle, SIGNAL(valueChanged(Vector3F)),
            this, SLOT(setJointOrient(Vector3F)));
			
	m_activeJoint = 0;
}

SkeletonJointEdit::~SkeletonJointEdit() {}

void SkeletonJointEdit::attachToJoint(int idx)
{
	if(!m_skeleton) return;
	m_activeJoint = m_skeleton->joint(idx);
	
    controlGrp->setTitle(tr(m_activeJoint->particalName().c_str()));
	
	updateValues();
}

void SkeletonJointEdit::setJointTranslation(Vector3F v)
{
	if(!m_activeJoint) return;
	m_activeJoint->setTranslation(v);
	emit valueChanged();
}

void SkeletonJointEdit::setJointRotation(Vector3F v)
{
	if(!m_activeJoint) return;
	m_activeJoint->setRotationAngles(v);
	emit valueChanged();
}

void SkeletonJointEdit::setJointOrient(Vector3F v)
{
	if(!m_activeJoint) return;
	m_activeJoint->setJointOrient(v);
	emit valueChanged();
}

void SkeletonJointEdit::updateValues()
{
	if(!m_activeJoint) return;
	translation->setValue(m_activeJoint->translation());
	rotateAngle->setValue(m_activeJoint->rotationAngles());
	rotateAngle->setDOF(m_activeJoint->rotateDOF());
	orientAngle->setValue(m_activeJoint->jointOrient());
}
