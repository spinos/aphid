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
#include <BaseBrush.h>
#include <KdTreeDrawer.h>
#include <BezierDrawer.h>
#include <MlDrawer.h>
#include <BezierCurve.h>
#include <ToolContext.h>
#include <bezierPatch.h>
#include <MlFeather.h>
#include <MlSkin.h>
#include <BakeDeformer.h>
#include <PlaybackControl.h>
#include <zEXRImage.h>
#include <SelectCondition.h>
#include <FloodCondition.h>
#include "MlCalamus.h"
#include <BaseCamera.h>
#include <BaseLight.h>
#include "MlEngine.h"
#include <TransformManipulator.h>

GLWidget::GLWidget(QWidget *parent) : ScenePort(parent)
{
	std::cout<<"3Dview ";
	m_bezierDrawer = new BezierDrawer;
	m_featherDrawer = new MlDrawer;
	m_featherDrawer->create("mallard.mlc");
	m_engine = new MlEngine(m_featherDrawer);
	MlCalamus::FeatherLibrary = this;
	m_featherTexId = m_featherDistrId = -1;
	cleanSheet();
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
	delete m_featherDrawer;
	delete m_bezierDrawer;
	delete m_engine;
}

void GLWidget::clientDraw()
{
	getDrawer()->m_surfaceProfile.apply();
	
	getDrawer()->setColor(0.37f, .59f, .9f);
	
	//if(m_featherDistrId > -1){
	if(0){
		getDrawer()->setColor(.8f, .8f, .8f);
		getDrawer()->bindTexture(m_featherDistrId);
	}

	m_bezierDrawer->drawBuffer(selectedTexture());
	m_bezierDrawer->unbindTexture();
	
	showActiveFaces();
	
	drawFeather();
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.2f, .8f, .4f);
	getDrawer()->drawLineBuffer(skin());
	
	showLights();
	showBrush();
	showManipulator();
}

void GLWidget::clientSelect(QMouseEvent *event)
{
    Ray ray = *getIncidentRay();
	switch (interactMode()) {
	    case ToolContext::CreateBodyContourFeather :
	        hitTest();
	        floodFeather();
	        break;
	    case ToolContext::EraseBodyContourFeather :
	        hitTest();
	        selectFeather();
	        m_featherDrawer->hideActive();
	        break;
	    case ToolContext::SelectByColor :
	        hitTest();
	        selectRegion();
	        break;
        case ToolContext::CombBodyContourFeather :
        case ToolContext::ScaleBodyContourFeather :
        case ToolContext::PitchBodyContourFeather :
		case ToolContext::Deintersect :
            hitTest();
            skin()->discardActive();
            selectFeather();
           break;
		case ToolContext::MoveTransform :
			selectLight(ray);
			break;
		case ToolContext::RotateTransform :
			selectLight(ray);
			break;
	    default:
			break;
	}
	ManipulateView::clientSelect(event);
}

void GLWidget::clientMouseInput(QMouseEvent *event)
{
    Ray ray = *getIncidentRay();
	switch (interactMode()) {
	    case ToolContext::CreateBodyContourFeather :
	        brush()->setToeByIntersect(&ray);
	        skin()->growFeather(brush()->toeDisplacement());
	        m_featherDrawer->updateActive();
	        break;
	    case ToolContext::EraseBodyContourFeather :
	        hitTest();
	        selectFeather();
	        m_featherDrawer->hideActive();
	        break;
	    case ToolContext::SelectByColor :
	        hitTest();
	        selectRegion();
	        break;
        case ToolContext::CombBodyContourFeather :
            brush()->setToeByIntersect(&ray);
            skin()->combFeather(brush()->toeDisplacement());
            m_featherDrawer->updateActive();
		    break;
        case ToolContext::ScaleBodyContourFeather :
            brush()->setToeByIntersect(&ray);
            skin()->scaleFeather(brush()->toeDisplacementDelta());
            m_featherDrawer->updateActive();
            break;
        case ToolContext::PitchBodyContourFeather :
            brush()->setToeByIntersect(&ray);
            skin()->pitchFeather(brush()->toeDisplacementDelta());
            m_featherDrawer->updateActive();
            break;
		case ToolContext::Deintersect :
			skin()->smoothShell(brush()->heelPosition(), brush()->getRadius(), brush()->strength());
            m_featherDrawer->updateActive();
			break;
		case ToolContext::MoveTransform :
		case ToolContext::RotateTransform :
			setDirty();
			break;
	    default:
			break;
	}
	ManipulateView::clientMouseInput(event);
}

char GLWidget::selectFeather()
{
	if(!ScenePort::selectFeather()) return 0;
	m_featherDrawer->clearCached();
	return 1;
}

char GLWidget::floodFeather()
{
	if(!ScenePort::floodFeather()) return 0;
	m_featherDrawer->addToBuffer();
	m_featherDrawer->clearCached();
	return 1;
}

void GLWidget::rebuildFeather()
{
	m_featherDrawer->disable();
	m_featherDrawer->initializeBuffer();
	m_featherDrawer->computeBufferIndirection();
	m_featherDrawer->rebuildBuffer();
	update();
	setDirty();
}

void GLWidget::bakeFrames()
{
	if(!playback()->isEnabled()) return;
	m_featherDrawer->clearCached();
	m_featherDrawer->enable();
	
	const int bakeMin = playback()->playbackMin();
	const int bakeMax = playback()->playbackMax();
	QProgressDialog * progress = new QProgressDialog(QString("Min %1 Max %2 Current %3").arg(bakeMin).arg(bakeMax).arg(bakeMin), tr("Cancel"), bakeMin, bakeMax, this);
	progress->setWindowModality(Qt::WindowModal);
	progress->setWindowTitle(tr("Baking Frames"));
	progress->show();
	progress->setValue(bakeMin);

	int i;
	for(i = bakeMin; i <= bakeMax; i++) {
	    if(progress->wasCanceled()) {
	        progress->close();
	        break;
	    }
	    
		updateOnFrame(i);
		
		progress->setLabelText(QString("Min %1 Max %2 Current %3").arg(bakeMin).arg(bakeMax).arg(i));
		progress->setValue(i);
	}

	delete progress;
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
	m_featherDrawer->initializeBuffer();
}

void GLWidget::doClear()
{
	ManipulateView::clearSelection();
	MlScene::doClear();
	if(m_featherTexId > -1) {
		getDrawer()->clearTexture(m_featherTexId);
		m_featherTexId = -1;
	}
	if(m_featherDistrId > -1) {
		getDrawer()->clearTexture(m_featherDistrId);
		m_featherDistrId = -1;
	}
	m_bezierDrawer->clearBuffer();
	m_featherDrawer->initializeBuffer();
	emit sceneNameChanged(tr("untitled"));
	update();
}

void GLWidget::doClose()
{
	if(isClosing()) m_featherDrawer->close();
	MlScene::doClose();
}

void GLWidget::exportBake()
{   
    if(skin()->numFeathers() < 1) {
        QMessageBox::information(this, tr("Warning"),
                                    tr("No feather to export."));
        return;
    }
    
    if(!playback()->isEnabled()) {
        QMessageBox::information(this, tr("Warning"),
                                    tr("No animation to export."));
        return;   
    }
    
    if(!m_featherDrawer->isBaked(playback()->rangeLength())) {
        QMessageBox::information(this, tr("Warning"),
                                    tr("Animation not fully baked, cannot export."));
        return; 
    }
	
	if(isDirty()) {
		QMessageBox::information(this, tr("Warning"),
                                    tr("Current scene has unsaved changes, save before export."));
        return; 
	}
    
    QString selectedFilter;
	QString fileName = QFileDialog::getSaveFileName(this,
							tr("Export Bake to File"),
							tr("info"),
							tr("All Files (*);;Feather Bake Files (*.mlb)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName == "") return;
	m_featherDrawer->setSceneName(MlScene::fileName());
	int rangeMin, rangeMax;
	bakeRange(rangeMin, rangeMax);
	m_featherDrawer->setBakeRange(rangeMin, rangeMax);
	if(m_featherDrawer->doCopy(fileName.toUtf8().data()))
		emit sendMessage(QString("Exported feather cache to file %1").arg(fileName));
}

void GLWidget::updateOnFrame(int x)
{
	if(!playback()->isEnabled()) return;
	clearFocus();
	
	if(!deformBody(x)) return;
	
	m_featherDrawer->setCurrentFrame(x);
	m_featherDrawer->setCurrentOrigin(bodyDeformer()->frameCenter());
	setCollision(skin());
	m_featherDrawer->rebuildBuffer();
	m_bezierDrawer->updateBuffer(body());
	
	update();
	
	setRebuildTree();
}

void GLWidget::afterOpen()
{
	MlScene::afterOpen();
	
	m_featherDrawer->clearCached();
	m_featherDrawer->setSkin(skin());
	m_featherDrawer->computeBufferIndirection();
	m_featherDrawer->rebuildBuffer();
	m_bezierDrawer->rebuildBuffer(body());
	buildTree();
	std::string febkgrd = featherEditBackground();
	if(febkgrd != "unknown") {
		setFeatherTexture(febkgrd);
		emit sendFeatherEditBackground(tr(febkgrd.c_str()));
	}
	std::string fedistr = featherDistributionMap();
	if(fedistr != "unknown")
		loadFeatherDistribution(fedistr);
	if(numLights() < 1) defaultLighting();
	emit sceneOpened();
}

void GLWidget::testRender()
{
	const QSize sz(renderImageWidth(), renderImageHeight());
	emit renderResChanged(sz);
	QString engineName(m_engine->rendererName().c_str());
	emit renderEngineChanged(engineName);
	emit renderStarted(renderName());
	prepareRender();
	m_engine->setLights(this);
	m_engine->setOptions(this);
	m_engine->preRender();
}

void GLWidget::receiveCancelRender()
{
	m_engine->interruptRender();
}

void GLWidget::importBody(const std::string & fileName)
{
	ScenePort::importBody(fileName);
	m_featherDrawer->initializeBuffer();
	m_bezierDrawer->rebuildBuffer(body());
	m_featherDrawer->setSkin(skin());
	buildTree();
	update();
}

QString GLWidget::renderName() const
{
	return QString("take_%1").arg(rand());
}

void GLWidget::drawFeather()
{
	if(!shouldDisplayFeather()) return;
	getDrawer()->m_surfaceProfile.apply();
	if(m_featherTexId > -1) {
		getDrawer()->setColor(.8f, .8f, .8f);
		getDrawer()->bindTexture(m_featherTexId);
	}
	else 
		getDrawer()->setColor(0.f, .71f, .51f);
	
	m_featherDrawer->draw();
	m_featherDrawer->unbindTexture();
}
//:~
