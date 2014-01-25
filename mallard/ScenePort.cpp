/*
 *  ScenePort.cpp
 *  mallard
 *
 *  Created by jian zhang on 1/23/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include "ScenePort.h"
#include "MlSkin.h"
#include "MlEngine.h"
#include <BaseCamera.h>
#include <KdTreeDrawer.h>
#include <ToolContext.h>
#include <TransformManipulator.h>
#include <AccPatchMesh.h>
#include <BaseBrush.h>
#include <zEXRImage.h>
#include <SelectCondition.h>
#include <FloodCondition.h>

ScenePort::ScenePort(QWidget *parent) : ManipulateView(parent) 
{
	getIntersectionContext()->setComponentFilterType(PrimitiveFilter::TFace);
	perspCamera()->setNearClipPlane(1.f);
	perspCamera()->setFarClipPlane(1000.f);
	usePerspCamera();
	setRenderCamera(getCamera());
	m_displayFeather = true;
}

ScenePort::~ScenePort() {}

PatchMesh * ScenePort::activeMesh() const
{
	return static_cast<PatchMesh *>(body());
}

void ScenePort::finishCreateFeather()
{
	skin()->finishCreateFeather();
}

void ScenePort::finishEraseFeather()
{
	skin()->finishEraseFeather();
}

void ScenePort::deselectFeather()
{
	skin()->discardActive();
}

void ScenePort::receiveCameraChanged()
{
	updatePerspProjection();
	update();
}

void ScenePort::receiveBarbChanged()
{
	setDirty();
}

void ScenePort::importFeatherDistributionMap()
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

void ScenePort::importBodyMesh()
{
	QString temQStr = QFileDialog::getOpenFileName(this, 
		tr("Open A Model File As Skin"), "../", tr("Mesh(*.m)"));
	
	if(temQStr == NULL)
		return;
		
	importBody(temQStr.toStdString());
}

void ScenePort::receiveFeatherEditBackground(QString name)
{
	setFeatherTexture(name.toUtf8().data());
	update();
}

void ScenePort::receiveFeatherAdded()
{
	setCollision(skin());
}

void ScenePort::showLights() const
{
	getDrawer()->drawLights(*this);
	if(interactMode() == ToolContext::MoveTransform || interactMode() == ToolContext::RotateTransform) {
		for(unsigned i = 0; i < numLights(); i++) getDrawer()->transform(getLight(i));
	}
}

char ScenePort::selectLight(const Ray & incident)
{
	if(!LightGroup::selectLight(incident)) return 0;

	std::clog<<"selected "<<selectedLight()->name()<<"\n";
	manipulator()->attachTo(selectedLight());
	return 1;
}

void ScenePort::setUseDisplaySize(bool x)
{
	if(x) {
		const QSize sz = size();
		setRenderImageWidth(sz.width());
		setRenderImageHeight(sz.height());
	}
	RenderOptions::setUseDisplaySize(x);
}

void ScenePort::clientDeselect(QMouseEvent *event)
{
    if(interactMode() == ToolContext::CreateBodyContourFeather)
		finishCreateFeather();

	ManipulateView::clientDeselect(event);
}

bool ScenePort::selectFeatherExample(unsigned x)
{
	bool r = MlFeatherCollection::selectFeatherExample(x);
	if(r) emit featherSelectionChanged();
	return r;
}

void ScenePort::resizeEvent( QResizeEvent * event )
{
	if(useDisplaySize()) {
		const QSize sz = event->size();
		setRenderImageWidth(sz.width());
		setRenderImageHeight(sz.height());
	}
	ManipulateView::resizeEvent(event);
}

void ScenePort::focusOutEvent(QFocusEvent * event)
{
	if(interactMode() == ToolContext::EraseBodyContourFeather)
		finishEraseFeather();
	deselectFeather();
	ManipulateView::focusOutEvent(event);
}

void ScenePort::clearSelection()
{
	skin()->clearCollisionRegion();
	skin()->clearBuffer();
	skin()->clearActiveRegion();
	ManipulateView::clearSelection();
}

void ScenePort::setFeatherTexture(const std::string & name)
{
	ZEXRImage image;
	if(!image.open(name)) return;
	image.verbose();
	makeCurrent();
	m_featherTexId = getDrawer()->loadTexture(m_featherTexId, &image);
	doneCurrent();
	MlScene::setFeatherTexture(name);
}

void ScenePort::loadFeatherDistribution(const std::string & name)
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

void ScenePort::importBody(const std::string & fileName)
{
	ManipulateView::clearSelection();
	MlScene::importBody(fileName);
	setDirty();
	update();
}

void ScenePort::selectRegion()
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

void ScenePort::chooseBake()
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

void ScenePort::beforeSave()
{
	if(interactMode() == ToolContext::EraseBodyContourFeather)
		finishEraseFeather();	
	deselectFeather();
}

void ScenePort::cleanSheet()
{
	clearSelection();
	clear();
	defaultLighting();
	emit sceneOpened();
}

void ScenePort::saveSheet()
{	
	if(MlScene::save())
		emit sendMessage(QString("Scene file %1 is saved").arg(MlScene::fileName().c_str()));
}

void ScenePort::saveSheetAs()
{
	if(MlScene::saveAs("")) {
		QString s(MlScene::fileName().c_str());
		emit sceneNameChanged(s);
		emit sendMessage(QString("Saved scene file as %1").arg(s));
	}
}

QString ScenePort::openSheet(QString name)
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

void ScenePort::revertSheet()
{
	if(revert())
		emit sendMessage(QString("Scene file %1 is reverted to saved").arg(tr(MlScene::fileName().c_str())));
}

bool ScenePort::confirmDiscardChanges()
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

std::string ScenePort::chooseOpenFileName()
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

std::string ScenePort::chooseSaveFileName()
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

char ScenePort::selectFeather()
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return 0;
	
	selectFaces();
	
	SelectCondition condition;
	condition.setCenter(ctx->m_hitP);
	condition.setNormal(ctx->m_hitN);
	condition.setMaxDistance(brush()->getRadius());
	condition.setProbability(brush()->strength());
	condition.setRegionFilter(brush()->filterByColor());
	
	skin()->select(selectedQue(), &condition);
	setDirty();
	
	return 1;
}

char ScenePort::floodFeather()
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return 0;
	
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
	
	if(brush()->filterByColor() && skin()->hasActiveRegion()) {
		skin()->restFloodFacesAsActive();
		condition.setDistanceFilter(0);
	}
	else {
		selectFaces();
		skin()->resetCollisionRegion(selectedQue());
		skin()->resetFloodFaces();
	}
	
	MlCalamus ac;
	ac.setFeatherId(selectedFeatherExampleId());
	ac.setRotateY(brush()->getPitch());
	
	skin()->floodAround(ac, &condition);
	setDirty();
	return 1;
}

void ScenePort::tagFace(bool x)
{
	BaseMesh * m = body();
	char * g = m->perFaceTag("growon");
	std::deque<unsigned>::const_iterator it = selectedQue().begin();
	for(; it != selectedQue().end(); ++it) g[*it] = x;
	updateFaceTagMap(selectedQue(), g);
	setDirty();
	selectTexture(GrowOnTag);
	update();
}

bool ScenePort::shouldDisplayFeather() const { return m_displayFeather; }

void ScenePort::setDisplayFeather(bool x) { m_displayFeather = x; update(); }

void ScenePort::selectMap(int i)
{
	selectTexture(i);
	update();
}

#include <PointInsidePolygonTest.h>
void ScenePort::testPatch()
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

void ScenePort::testCurvature()
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