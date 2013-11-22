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
#include <BaseBrush.h>
#include <KdTreeDrawer.h>
#include <BezierDrawer.h>
#include <MlDrawer.h>
#include <BezierCurve.h>
#include <ToolContext.h>
#include <bezierPatch.h>
#include <InverseBilinearInterpolate.h>
#include <MlFeather.h>
#include <MlSkin.h>
#include <BakeDeformer.h>
#include <PlaybackControl.h>
#include <zEXRImage.h>
#include <SelectCondition.h>
#include "MlCalamus.h"

GLWidget::GLWidget(QWidget *parent) : SingleModelView(parent)
{
	std::cout<<"3Dview ";
	m_bezierDrawer = new BezierDrawer;
	m_featherDrawer = new MlDrawer;
	m_featherDrawer->create("mallard.mlc");
	MlCalamus::FeatherLibrary = this;
	getIntersectionContext()->setComponentFilterType(PrimitiveFilter::TFace);
	m_featherTexId = m_featherDistrId = -1;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientDraw()
{
	//getDrawer()->setGrey(1.f);
	//getDrawer()->edge(mesh());
	getDrawer()->m_surfaceProfile.apply();
	
	//getDrawer()->drawKdTree(getTree());
	
	if(m_featherTexId > -1) {
		getDrawer()->setColor(.8f, .8f, .8f);
		getDrawer()->bindTexture(m_featherTexId);
	}
	else 
		getDrawer()->setColor(0.f, .71f, .51f);
	
	m_featherDrawer->draw(skin());
	m_featherDrawer->unbindTexture();
	
	getDrawer()->setColor(0.37f, .59f, .9f);
	
	if(m_featherDistrId > -1){
		getDrawer()->setColor(.8f, .8f, .8f);
		getDrawer()->bindTexture(m_featherDistrId);
	}

	m_bezierDrawer->drawBuffer();
	m_bezierDrawer->unbindTexture();
	
	getDrawer()->m_wireProfile.apply();
	getDrawer()->setColor(.2f, .8f, .4f);
	getDrawer()->drawLineBuffer(skin());
	
	//drawSelection();
	showBrush();
}

void GLWidget::loadMesh(std::string filename)
{
	skin()->cleanup();
	m_featherDrawer->initializeBuffer();
	disableDeformer();
	ESMUtil::ImportPatch(filename.c_str(), mesh());
	afterOpen();
	setDirty();
	update();
}

void GLWidget::clientSelect()
{
    Vector3F hit;
	Ray ray = *getIncidentRay();
	switch (interactMode()) {
	    case ToolContext::CreateBodyContourFeather :
	        hitTest(ray, hit);
	        floodFeather();
	        break;
	    case ToolContext::EraseBodyContourFeather :
	        hitTest(ray, hit);
	        selectFeather(skin()->eraseRegion());
	        m_featherDrawer->hideActive(skin());
	        break;
	    case ToolContext::SelectByColor :
	        hitTest(ray, hit);
	        selectRegion();
	        break;
        case ToolContext::CombBodyContourFeather :
        case ToolContext::ScaleBodyContourFeather :
        case ToolContext::PitchBodyContourFeather :
            hitTest(ray, hit);
            skin()->discardActive();
            selectFeather();
           break;
	    default:
			break;
	}
}

void GLWidget::clientMouseInput()
{
    Vector3F hit;
	Ray ray = *getIncidentRay();
	switch (interactMode()) {
	    case ToolContext::CreateBodyContourFeather :
	        brush()->setToeByIntersect(&ray);
	        skin()->growFeather(brush()->toeDisplacement());
	        m_featherDrawer->updateActive(skin());
	        break;
	    case ToolContext::EraseBodyContourFeather :
	        hitTest(ray, hit);
	        selectFeather(skin()->eraseRegion());
	        m_featherDrawer->hideActive(skin());
	        break;
	    case ToolContext::SelectByColor :
	        hitTest(ray, hit);
	        selectRegion();
	        break;
        case ToolContext::CombBodyContourFeather :
            brush()->setToeByIntersect(&ray);
            skin()->combFeather(brush()->toeDisplacement(), brush()->heelPosition(), brush()->getRadius());
            m_featherDrawer->updateActive(skin());
		    break;
        case ToolContext::ScaleBodyContourFeather :
            brush()->setToeByIntersect(&ray);
            skin()->scaleFeather(brush()->toeDisplacementDelta(), brush()->heelPosition(), brush()->getRadius());
            m_featherDrawer->updateActive(skin());
            break;
        case ToolContext::PitchBodyContourFeather :
            brush()->setToeByIntersect(&ray);
            skin()->pitchFeather(brush()->toeDisplacementDelta(), brush()->heelPosition(), brush()->getRadius());
            m_featherDrawer->updateActive(skin());
            break;
	    default:
			break;
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

void GLWidget::selectFeather(char byRegion)
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return;
	
	brush()->setSpace(ctx->m_hitP, ctx->m_hitN);
	brush()->resetToe();
	
	SelectCondition condition;
	condition.setCenter(ctx->m_hitP);
	condition.setNormal(ctx->m_hitN);
	condition.setMaxDistance(brush()->getRadius());
	
	if(interactMode() == ToolContext::EraseBodyContourFeather) condition.setProbability(brush()->strength());
	else condition.setProbability(1.f);
	
	condition.setRegionFilter(byRegion);
	
	skin()->selectAround(ctx->m_componentIdx, &condition);
	m_featherDrawer->clearCached();
	setDirty();
}

void GLWidget::selectRegion()
{
    if(m_featherDistrId < 0) return;
    IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) {
		clearSelection();
		return;
    }
    skin()->selectRegion(ctx->m_componentIdx, ctx->m_patchUV);
	skin()->resetRegionFaces();
	skin()->setFloodCondition(MlSkin::ByColor);
}

void GLWidget::floodFeather()
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return;
	
	brush()->setSpace(ctx->m_hitP, ctx->m_hitN);
	brush()->resetToe();
	
	//Vector3F rr = getIncidentRay()->m_dir;
	//rr.reverse();
	//if(rr.dot(brush()->normal()) < .34f) return;
	
	MlCalamus ac;
	ac.setFeatherId(selectedFeatherExampleId());
	ac.setRotateY(brush()->getPitch());
	
	if(skin()->floodRegion() && skin()->floodCondition() == MlSkin::ByColor) {
		skin()->restFloodFacesAsActive();
	}
	else {
		skin()->resetCollisionRegionAround(ctx->m_componentIdx, ctx->m_hitP, brush()->getRadius());
		skin()->resetFloodFaces();
	}
	skin()->floodAround(ac, ctx->m_hitP, ctx->m_hitN, brush()->getRadius(), brush()->minDartDistance());
	m_featherDrawer->addToBuffer(skin());
	m_featherDrawer->clearCached();
	setDirty();
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
	m_featherDrawer->initializeBuffer();
	m_featherDrawer->computeBufferIndirection(skin());
	m_featherDrawer->rebuildBuffer(skin(), true);
	update();
	setDirty();
}

void GLWidget::bakeFrames()
{
	if(!playback()->isEnabled()) return;
	
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

void GLWidget::cleanSheet()
{
	clear();
}

bool GLWidget::confirmDiscardChanges()
{
	QMessageBox::StandardButton reply;
	if(isReverting()) {
		reply = QMessageBox::question(this, tr(" "),
                                    tr("Revert to latest saved version of the scene?"),
                                    QMessageBox::Yes | QMessageBox::Cancel);
	}
	else if(isClosing()) {
		reply = QMessageBox::question(this, tr(" "),
                                    tr("Save changes to the scene before closing?"),
                                    QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
		if (reply == QMessageBox::Yes)
			MlScene::save();
	}
	else {
		reply = QMessageBox::question(this, tr(" "),
                                    tr("Save changes to the scene before creating a new one?"),
                                    QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
    
		if (reply == QMessageBox::Yes)
			MlScene::save();
	}
	
	if(reply == QMessageBox::Cancel) return false;

	return true;
}

std::string GLWidget::chooseOpenFileName()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open scene from file"),
							tr("info"),
							tr("All Files (*);;Mallard Files (*.mal)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	return fileName.toUtf8().data();
}

std::string GLWidget::chooseSaveFileName()
{
	QString selectedFilter;
	QString fileName = QFileDialog::getSaveFileName(this,
							tr("Save scene to file"),
							tr("info"),
							tr("All Files (*);;Mallard Files (*.mal)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	return fileName.toUtf8().data();
}

void GLWidget::doClear()
{
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
	clearTree();
	clearTopology();
	emit sceneNameChanged(tr("untitled"));
	update();
}

void GLWidget::doClose()
{
	if(isClosing()) m_featherDrawer->close();
	MlScene::doClose();
}

void GLWidget::beforeSave()
{
	if(interactMode() == ToolContext::EraseBodyContourFeather)
		finishEraseFeather();
		
	deselectFeather();
}

void GLWidget::saveSheet()
{	
	if(MlScene::save())
		emit sendMessage(QString("Scene file %1 is saved").arg(MlScene::fileName().c_str()));
}

void GLWidget::saveSheetAs()
{
	if(MlScene::saveAs("")) {
		QString s(MlScene::fileName().c_str());
		emit sceneNameChanged(s);
		emit sendMessage(QString("Saved scene file as %1").arg(s));
	}
}

QString GLWidget::openSheet(QString name)
{
	if(name == tr("")) {
		if(!MlScene::open()) return tr("");
	}
	else {
		if(!MlScene::open(name.toUtf8().data())) return tr("");
	}
	
	QString res = tr(MlScene::fileName().c_str());
	
	emit sceneNameChanged(res);
	emit sendMessage(QString("Scene file %1 is opened").arg(res));
	return res;
}

void GLWidget::revertSheet()
{
	if(revert())
		emit sendMessage(QString("Scene file %1 is reverted to saved").arg(tr(MlScene::fileName().c_str())));
}

void GLWidget::receiveFeatherEditBackground(QString name)
{
	setFeatherTexture(name.toUtf8().data());
	update();
}

void GLWidget::chooseBake()
{
	if(body()->isEmpty()) {
		QMessageBox::information(this, tr("Warning"),
                                    tr("Mesh not loaded. Cannot attach bake."));
		return;
	}
	
	QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Load Body Animation from file"),
							tr("info"),
							tr("All Files (*);;Mesh Bake Files (*.h5)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName == "") return;
	if(readBakeFromFile(fileName.toUtf8().data()))
		emit sendMessage(QString("Attached body animation in file %1").arg(fileName));
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
    
    if(playback()->rangeLength() > m_featherDrawer->numCachedSlices("/p")) {
        QMessageBox::information(this, tr("Warning"),
                                    tr("Animation not fully baked, cannot export."));
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
	if(m_featherDrawer->doCopy(fileName.toUtf8().data()))
		emit sendMessage(QString("Exported feather cache to file %1").arg(fileName));
}

void GLWidget::updateOnFrame(int x)
{
	if(!playback()->isEnabled()) return;
	clearFocus();
	
	if(!deformBody(x)) return;
	
	body()->update(m_topo);
	m_featherDrawer->setCurrentFrame(x);
	m_featherDrawer->setCurrentOrigin(bodyDeformer()->frameCenter());
	m_featherDrawer->rebuildBuffer(skin());
	m_bezierDrawer->rebuildBuffer(body());
	
	update();
	
	setRebuildTree();
}

void GLWidget::afterOpen()
{
	body()->putIntoObjectSpace();
	buildTopology();
	body()->setup(m_topo);
	buildTree();
	skin()->setBodyMesh(body(), m_topo);
	skin()->computeFaceCalamusIndirection();
	bodyDeformer()->setMesh(body());
	m_featherDrawer->clearCached();
	m_featherDrawer->computeBufferIndirection(skin());
	m_featherDrawer->rebuildBuffer(skin(), true);
	m_bezierDrawer->rebuildBuffer(body());
	delayLoadBake();
	std::string febkgrd = featherEditBackground();
	if(febkgrd != "unknown") {
		setFeatherTexture(febkgrd);
		emit sendFeatherEditBackground(tr(febkgrd.c_str()));
	}
	std::string fedistr = featherDistributionMap();
	if(fedistr != "unknown")
		loadFeatherDistribution(fedistr);
}

void GLWidget::focusOutEvent(QFocusEvent * event)
{
	if(interactMode() == ToolContext::EraseBodyContourFeather)
		finishEraseFeather();
	deselectFeather();
	SingleModelView::focusOutEvent(event);
}

void GLWidget::setFeatherTexture(const std::string & name)
{
	ZEXRImage image;
	if(!image.open(name)) return;
	image.verbose();
	makeCurrent();
	m_featherTexId = getDrawer()->loadTexture(m_featherTexId, &image);
	doneCurrent();
	MlScene::setFeatherTexture(name);
}

void GLWidget::importFeatherDistributionMap()
{
    if(body()->isEmpty()) {
		QMessageBox::information(this, tr("Warning"),
                                    tr("Mesh not loaded. Cannot attach feather distribution map."));
		return;
	}
	
    QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open .exr image file as the Feather Distribution Map"),
							tr("info"),
							tr("All Files (*);;EXR Files (*.exr)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName != "") {
		loadFeatherDistribution(fileName.toUtf8().data());
	}
}

void GLWidget::loadFeatherDistribution(const std::string & name)
{
    ZEXRImage *image = new ZEXRImage;
	if(!image->open(name)) return;
	image->verbose();
	makeCurrent();
	m_featherDistrId = getDrawer()->loadTexture(m_featherDistrId, image);
	doneCurrent();
	skin()->setDistributionMap(image);
	setFeatherDistributionMap(name);
}

void GLWidget::receiveFloodRegion(int state)
{
	if(state == Qt::Unchecked)
		skin()->setFloodRegion(0);
	else
		skin()->setFloodRegion(1);
}

void GLWidget::receiveEraseRegion(int state)
{
    if(state == Qt::Unchecked)
		skin()->setEraseRegion(0);
	else
		skin()->setEraseRegion(1);
}

void GLWidget::clearSelection()
{
	skin()->setFloodCondition(MlSkin::ByDistance);
	skin()->clearCollisionRegion();
	skin()->clearBuffer();
	skin()->clearRegionFaces();
}
//:~
