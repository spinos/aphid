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

GLWidget::GLWidget(QWidget *parent) : ManipulateView(parent)
{
	std::cout<<"3Dview ";
	m_bezierDrawer = new BezierDrawer;
	m_featherDrawer = new MlDrawer;
	m_featherDrawer->create("mallard.mlc");
	m_engine = new MlEngine(m_featherDrawer);
	MlCalamus::FeatherLibrary = this;
	getIntersectionContext()->setComponentFilterType(PrimitiveFilter::TFace);
	m_featherTexId = m_featherDistrId = -1;
	m_floodByRegion = m_eraseByRegion = 0;
	perspCamera()->setNearClipPlane(1.f);
	perspCamera()->setFarClipPlane(100000.f);
	usePerspCamera();
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
	
	if(m_featherTexId > -1) {
		getDrawer()->setColor(.8f, .8f, .8f);
		getDrawer()->bindTexture(m_featherTexId);
	}
	else 
		getDrawer()->setColor(0.f, .71f, .51f);
	
	m_featherDrawer->draw();
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
	
	showLights();
	showBrush();
	showManipulator();
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
	        selectFeather(m_eraseByRegion);
	        m_featherDrawer->hideActive();
	        break;
	    case ToolContext::SelectByColor :
	        hitTest(ray, hit);
	        selectRegion();
	        break;
        case ToolContext::CombBodyContourFeather :
        case ToolContext::ScaleBodyContourFeather :
        case ToolContext::PitchBodyContourFeather :
		case ToolContext::Deintersect :
            hitTest(ray, hit);
            skin()->discardActive();
            selectFeather();
           break;
		case ToolContext::MoveTransform :
			manipulator()->setToMove();
			selectLight(ray);
			break;
		case ToolContext::RotateTransform :
			manipulator()->setToRotate();
			selectLight(ray);
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
	        m_featherDrawer->updateActive();
	        break;
	    case ToolContext::EraseBodyContourFeather :
	        hitTest(ray, hit);
	        selectFeather(m_eraseByRegion);
	        m_featherDrawer->hideActive();
	        break;
	    case ToolContext::SelectByColor :
	        hitTest(ray, hit);
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
			manipulator()->perform(&ray);
			setDirty();
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
	manipulator()->detach();
}

PatchMesh * GLWidget::activeMesh()
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
	skin()->resetActiveRegion();
}

void GLWidget::floodFeather()
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return;
	
	brush()->setSpace(ctx->m_hitP, ctx->m_hitN);
	brush()->resetToe();
	
	FloodCondition condition;
	condition.setCenter(ctx->m_hitP);
	condition.setNormal(ctx->m_hitN);
	condition.setMaxDistance(brush()->getRadius());
	condition.setMinDistance(brush()->minDartDistance());
	condition.setProbability(brush()->strength());
	condition.setDistanceFilter(1);
	
	if(skin()->hasActiveRegion()) condition.setRegionFilter(1);
	else condition.setRegionFilter(0);
	
	if(m_floodByRegion && skin()->hasActiveRegion()) {
		skin()->restFloodFacesAsActive();
		condition.setDistanceFilter(0);
	}
	else {
		skin()->resetCollisionRegionByDistance(ctx->m_componentIdx, ctx->m_hitP, brush()->getRadius());
		skin()->resetFloodFaces();
	}
	
	MlCalamus ac;
	ac.setFeatherId(selectedFeatherExampleId());
	ac.setRotateY(brush()->getPitch());
	
	skin()->floodAround(ac, &condition);
	m_featherDrawer->addToBuffer();
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

void GLWidget::cleanSheet()
{
	clear();
	defaultLighting();
	emit sceneOpened();
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
	return validateFileExtension(fileName.toUtf8().data());
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

void GLWidget::receiveFeatherAdded()
{
	setCollision(skin());
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
	m_bezierDrawer->rebuildBuffer(body());
	
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

void GLWidget::focusOutEvent(QFocusEvent * event)
{
	if(interactMode() == ToolContext::EraseBodyContourFeather)
		finishEraseFeather();
	deselectFeather();
	ManipulateView::focusOutEvent(event);
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
		m_floodByRegion = 0;
	else
		m_floodByRegion = 1;
}

void GLWidget::receiveEraseRegion(int state)
{
    if(state == Qt::Unchecked)
		m_eraseByRegion = 0;
	else
		m_eraseByRegion = 1;
}

void GLWidget::clearSelection()
{
	skin()->clearCollisionRegion();
	skin()->clearBuffer();
	skin()->clearActiveRegion();
}

void GLWidget::testCurvature()
{
    getDrawer()->setColor(.2f, .2f, .2f);
	getDrawer()->edge(activeMesh());
	IntersectionContext * ctx = getIntersectionContext();
    if(ctx->m_success) {
		std::vector<Vector3F> us;
		skin()->getClustering(ctx->m_componentIdx, us);
		getDrawer()->setColor(0.8f, .8f, .1f);
		getDrawer()->lines(us);
	}
	return;
	
	if(ctx->m_success) {
		Vector3F p = brush()->toePosition();
		Vector3F clsP, clsN;
		clsN = skin()->getClosestNormal(p, 1000.f, clsP);
		
		std::vector<Vector3F> us;
		us.push_back(p);
		us.push_back(clsP);
		us.push_back(clsP);
		us.push_back(clsP + clsN);
		
		getDrawer()->setColor(0.8f, .8f, .4f);
		getDrawer()->lines(us);
		
		us.clear();
		getDrawer()->setColor(0.2f, .8f, .1f);
		body()->getPatchHir(ctx->m_componentIdx, us);
		getDrawer()->lines(us);
	}

	if(body()->getNumVertices() > 1 /*&& interactMode() == ToolContext::Deintersect*/) {
		getDrawer()->setColor(1.f, 0.f, 0.f);
		getDrawer()->vertexNormal(body());
		getDrawer()->setColor(0.f, 1.f, 0.f);
		getDrawer()->perVertexVector(body(), "aftshell");
		getDrawer()->setColor(0.8f, .8f, .4f);
		std::vector<Vector3F> us;
		skin()->shellUp(us);
		getDrawer()->lines(us);
		
	}
	if(skin()->numRegionElements() < 1) return;
	Vector3F p = brush()->heelPosition();
	Vector3F n = brush()->normal();
	Vector3F t = brush()->toeDisplacement();
	Vector3F bn = t.cross(n);
	bn.normalize();
	t = n.cross(bn);
	t.normalize();
	Matrix33F m;
	m.fill(n, bn, t);
	getDrawer()->coordsys(m, 1.f, &p);
	
	Matrix33F m2, m1 = m;
	Vector3F d;
	Vector2F cvt;
	float b = 0.f;
	for(unsigned i = 0; i < 8; i++) {
		cvt = skin()->curvatureAt(m1, m2, p, .5f);
		getDrawer()->coordsys(m2, .5f, &p);
		m1 = m2;
		d = Vector3F::ZAxis;
		d = m2.transform(d);      
		p += d * .5f;  
		b += cvt.x;   
	} 
}

void GLWidget::resizeEvent( QResizeEvent * event )
{
	if(useDisplaySize()) {
		const QSize sz = event->size();
		setRenderImageWidth(sz.width());
		setRenderImageHeight(sz.height());
	}
	ManipulateView::resizeEvent(event);
}

void GLWidget::testRender()
{
	const QSize sz(renderImageWidth(), renderImageHeight());
	emit renderResChanged(sz);
	prepareRender();
	m_engine->setCamera(getCamera());
	m_engine->setLights(this);
	m_engine->setOptions(this);
	m_engine->preRender();
}

void GLWidget::receiveCancelRender()
{
	m_engine->interruptRender();
}

char GLWidget::selectLight(const Ray & incident)
{
	if(!LightGroup::selectLight(incident)) return 0;

	std::clog<<"selected "<<selectedLight()->name()<<"\n";
	manipulator()->attachTo(selectedLight());
	manipulator()->start(&incident);
	return 1;
}

void GLWidget::showLights() const
{
	getDrawer()->drawLights(*this);
	if(interactMode() == ToolContext::MoveTransform || interactMode() == ToolContext::RotateTransform) {
		for(unsigned i = 0; i < numLights(); i++) getDrawer()->transform(getLight(i));
	}
}

bool GLWidget::selectFeatherExample(unsigned x)
{
	bool r = MlFeatherCollection::selectFeatherExample(x);
	emit featherSelectionChanged();
	return r;
}

void GLWidget::receiveBarbChanged()
{
	setDirty();
}

void GLWidget::importBodyMesh()
{
	QString temQStr = QFileDialog::getOpenFileName(this, 
		tr("Open A Model File As Skin"), "../", tr("Mesh(*.m)"));
	
	if(temQStr == NULL)
		return;
		
	importBody(temQStr.toStdString());
}

void GLWidget::importBody(const std::string & fileName)
{
	MlScene::importBody(fileName);
	m_featherDrawer->initializeBuffer();
	m_bezierDrawer->rebuildBuffer(body());
	m_featherDrawer->setSkin(skin());
	buildTree();
	setDirty();
	update();
}

void GLWidget::setUseDisplaySize(bool x)
{
	if(x) {
		const QSize sz = size();
		setRenderImageWidth(sz.width());
		setRenderImageHeight(sz.height());
	}
	RenderOptions::setUseDisplaySize(x);
}
#include <PointInsidePolygonTest.h>
void GLWidget::testPatch()
{
	Vector3F a;
	Vector3F b(10.f, 2.f, 0.f);
	Vector3F c(10.f, 2.f, 0.f);
	Vector3F d(0.f, 1.f, -10.f);
	PointInsidePolygonTest pa(a, b, c, d);
	
	BaseLight *l = getLight(2);
	Vector3F ts = l->translation();
	Vector3F cs;
	char inside = 0;
	pa.distanceTo(ts, cs, inside);
	getDrawer()->quad(a, b, c, d);
	getDrawer()->arrow(pa.center(), pa.center() + pa.normal());
	getDrawer()->arrow(ts, cs);
}
//:~
