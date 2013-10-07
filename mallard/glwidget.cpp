/****************************************************************************
**
** Copyright (C) 2010 Nokia Corporation and/or its subsidiary(-ies).
** All rights reserved.
** Contact: Nokia Corporation (qt-info@nokia.com)
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial Usage
** Licensees holding valid Qt Commercial licenses may use this file in
** accordance with the Qt Commercial License Agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and Nokia.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 as published by the Free Software
** Foundation and appearing in the file LICENSE.LGPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU Lesser General Public License version 2.1 requirements
** will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** In addition, as a special exception, Nokia gives you certain additional
** rights.  These rights are described in the Nokia Qt LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
** If you have questions regarding the use of this file, please contact
** Nokia at qt-info@nokia.com.
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QtGui>
#include <QtOpenGL>

#include <math.h>
#include "glwidget.h"
#include <AccPatchMesh.h>
#include <EasemodelUtil.h>
#include <BezierDrawer.h>
#include <MlDrawer.h>
#include <BezierCurve.h>
#include <ToolContext.h>
#include <bezierPatch.h>
#include <InverseBilinearInterpolate.h>
#include <MlFeather.h>
#include <MlSkin.h>
#include <BakeDeformer.h>
#include "MlCalamus.h"

GLWidget::GLWidget(QWidget *parent) : SingleModelView(parent)
{
	m_bezierDrawer = new BezierDrawer;
	m_featherDrawer = new MlDrawer;
	MlCalamus::FeatherLibrary = this;
	
	getIntersectionContext()->setComponentFilterType(PrimitiveFilter::TFace);
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	getDrawer()->setGrey(1.f);
	//getDrawer()->edge(mesh());
	getDrawer()->m_surfaceProfile.apply();
	getDrawer()->setColor(0.37f, .59f, .9f);
	m_bezierDrawer->drawBuffer();
	//getDrawer()->drawKdTree(getTree());
	getDrawer()->setColor(0.f, .71f, .51f);
	
	//getDrawer()->m_wireProfile.apply();
	m_featherDrawer->drawFeather(skin());

	//drawSelection();
	showBrush();
}

void GLWidget::loadMesh(std::string filename)
{
	ESMUtil::ImportPatch(filename.c_str(), mesh());
	postLoad();
	setDirty();
}

void GLWidget::clientSelect()
{
    Vector3F hit;
	Ray ray = *getIncidentRay();
	if(interactMode() == ToolContext::SelectVertex) {
		pickupComponent(ray, hit);
	}
	else if(interactMode() == ToolContext::CreateBodyContourFeather) {
		hitTest(ray, hit);
		floodFeather();
	}
	else if(interactMode() == ToolContext::EraseBodyContourFeather) {
		hitTest(ray, hit);
		selectFeather();
		m_featherDrawer->hideActive(skin());
	}
	else if(interactMode() == ToolContext::CombBodyContourFeather || interactMode() == ToolContext::ScaleBodyContourFeather || interactMode() == ToolContext::PitchBodyContourFeather) {
		hitTest(ray, hit);
		skin()->discardActive();
		selectFeather();
	}
	setDirty();
}

void GLWidget::clientMouseInput()
{
    Vector3F hit;
	Ray ray = *getIncidentRay();
	if(interactMode() == ToolContext::SelectVertex) {
		pickupComponent(ray, hit);
	}
	else if(interactMode() == ToolContext::CreateBodyContourFeather) {
		brush()->setToeByIntersect(&ray);
		skin()->growFeather(brush()->toeDisplacement());
		m_featherDrawer->updateActive(skin());
	}
	else if(interactMode() == ToolContext::EraseBodyContourFeather) {
		hitTest(ray, hit);
		selectFeather();
		m_featherDrawer->hideActive(skin());
	}
	else if(interactMode() == ToolContext::CombBodyContourFeather) {
		brush()->setToeByIntersect(&ray);
		skin()->combFeather(brush()->toeDisplacement(), brush()->heelPosition(), brush()->getRadius());
		m_featherDrawer->updateActive(skin());
	}
	else if(interactMode() == ToolContext::ScaleBodyContourFeather) {
		brush()->setToeByIntersect(&ray);
		skin()->scaleFeather(brush()->toeDisplacementDelta(), brush()->heelPosition(), brush()->getRadius());
		m_featherDrawer->updateActive(skin());
	}
	else if(interactMode() == ToolContext::PitchBodyContourFeather) {
		brush()->setToeByIntersect(&ray);
		skin()->pitchFeather(brush()->toeDisplacementDelta(), brush()->heelPosition(), brush()->getRadius());
		m_featherDrawer->updateActive(skin());
	}
}

void GLWidget::clientDeselect()
{
    if(interactMode() == ToolContext::CreateBodyContourFeather) {
		skin()->finishCreateFeather();
		skin()->discardActive();
	}
}

PatchMesh * GLWidget::mesh()
{
	return body();
}

void GLWidget::selectFeather()
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return;
	
	brush()->setSpace(ctx->m_hitP, ctx->m_hitN);
	brush()->resetToe();
	
	skin()->selectAround(ctx->m_componentIdx, ctx->m_hitP, ctx->m_hitN, brush()->getRadius());
}

void GLWidget::floodFeather()
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return;
	
	brush()->setSpace(ctx->m_hitP, ctx->m_hitN);
	brush()->resetToe();
	
	Vector3F rr = getIncidentRay()->m_dir;
	rr.reverse();
	if(rr.dot(brush()->normal()) < .34f) return;
	
	const unsigned iface = ctx->m_componentIdx;
	
	MlCalamus ac;
	ac.setFeatherId(selectedFeatherExampleId());
	ac.setRotateY(brush()->getPitch());
	skin()->floodAround(ac, iface, ctx->m_hitP, ctx->m_hitN, brush()->getRadius(), brush()->minDartDistance());
	m_featherDrawer->addToBuffer(skin());
}

void GLWidget::finishEraseFeather()
{
	skin()->finishEraseFeather();
}

void GLWidget::deselectFeather()
{
	skin()->discardActive();
}

void GLWidget::rebuildFeather()
{
	m_featherDrawer->clearBuffer();
	m_featherDrawer->initializeBuffer();
	m_featherDrawer->rebuildBuffer(skin());
}

void GLWidget::clearFeather()
{
	QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, tr(" "),
                                    tr("Delete all feathers?"),
                                    QMessageBox::Yes | QMessageBox::Cancel);
    if (reply == QMessageBox::Cancel)
		return;
		
	skin()->clearFeather();
	m_featherDrawer->clearBuffer();
	m_featherDrawer->initializeBuffer();
}

void GLWidget::cleanSheet()
{
	newScene();
}

bool GLWidget::discardConfirm()
{
	QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, tr(" "),
                                    tr("Save changes to the scene before creating a new one?"),
                                    QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
    if (reply == QMessageBox::Cancel)
		return false;
	
	if (reply == QMessageBox::Yes)
		saveSheet();

	return true;
}

void GLWidget::clearScene()
{
	MlScene::clearScene();
	m_bezierDrawer->clearBuffer();
	m_featherDrawer->clearBuffer();
	m_featherDrawer->initializeBuffer();
	clearTree();
	clearTopology();
	emit sceneNameChanged(tr("untitled"));
	update();
}

void GLWidget::saveSheet()
{
	if(!shouldSave()) {
		qDebug()<<"Nothing to save.";
		return;
	}
	
	if(interactMode() == ToolContext::EraseBodyContourFeather)
		finishEraseFeather();
		
	deselectFeather();
	
	if(isUntitled()) saveSheetAs();
	else if(saveScene())
		emit sendMessage(QString("Scene file %1 is saved").arg(fileName().c_str()));
}

void GLWidget::saveSheetAs()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getSaveFileName(this,
							tr("Save scene to file"),
							tr("info"),
							tr("All Files (*);;Text Files (*.txt)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName != "") {
		saveSceneAs(fileName.toUtf8().data());
		emit sceneNameChanged(fileName);
		emit sendMessage(QString("Scene file %1 is saved").arg(fileName));
	}
}

void GLWidget::openSheet()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open scene from file"),
							tr("info"),
							tr("All Files (*);;Text Files (*.txt)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName != "") {
		openScene(fileName.toUtf8().data());
		postLoad();
		emit sceneNameChanged(fileName);
	}
}

void GLWidget::revertSheet()
{
	if(isUntitled()) return;
	QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, tr(" "),
                                    tr("Do you want to revert to the most recently saved version?"),
                                    QMessageBox::Yes | QMessageBox::Cancel);
    if (reply == QMessageBox::Cancel)
		return;
	revertScene();
	postLoad();
}

void GLWidget::receiveFeatherEditBackground(QString name)
{
	setFeatherEditBackground(name.toUtf8().data());
}

void GLWidget::postLoad()
{
	buildTopology();
	body()->setup(m_topo);
	buildTree();
	skin()->setBodyMesh(body(), m_topo);
	skin()->finishCreateFeather();
	bodyDeformer()->setMesh(body());
	m_bezierDrawer->rebuildBuffer(body());
	m_featherDrawer->rebuildBuffer(skin());
	update();
	std::string febkgrd = featherEditBackground();
	if(febkgrd != "unknown") emit sendFeatherEditBackground(tr(febkgrd.c_str()));
	emit sceneNameChanged(tr(fileName().c_str()));
	emit sendMessage(QString("Scene file %1 is loaded").arg(tr(fileName().c_str())));
}
//:~
