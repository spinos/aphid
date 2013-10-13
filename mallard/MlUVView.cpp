/*
 *  MlUVView.cpp
 *  mallard
 *
 *  Created by jian zhang on 10/2/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlUVView.h"
#include <MlFeatherCollection.h>
#include <MlFeather.h>
#include <BezierCurve.h>
#include <QtGui>
#include <ToolContext.h>
#include <zEXRImage.h>
MlFeatherCollection * MlUVView::FeatherLibrary = 0;

MlUVView::MlUVView(QWidget *parent) : Base2DView(parent)
{
	std::cout<<"UVview ";
	m_activeId = m_texId = -1;
	m_selectedVert = 0;
	m_image = new ZEXRImage;
	m_image->allBlack();
}

MlUVView::~MlUVView()
{
}

void MlUVView::clientDraw()
{
	if(m_texId > -1) {
		glPushMatrix();
		glScalef(32.f, 32.f, 1.f);
		glTranslatef(0.f, 0.f, -10.f);
		getDrawer()->texture(m_texId);
		glPopMatrix();
	}
	
	if(!FeatherLibrary) return;
	MlFeather *f;
	for(f = FeatherLibrary->firstFeatherExample(); FeatherLibrary->hasFeatherExample(); f = FeatherLibrary->nextFeatherExample()) {
		if(!f) continue;
		drawFeather(f);
	}
	
	if(m_activeId < 0) return;
	
	f = FeatherLibrary->selectedFeatherExample();
	if(!f) return;
	getDrawer()->boundingRectangle(f->getBoundingRectangle());
	
	if(interactMode() == ToolContext::MoveVertexInUV) {
		drawControlVectors(f);
		if(m_selectedVert) {
			Vector3F pv(m_selectVertWP);
			getDrawer()->cube(pv, 0.1f);
		}
	}
}

void MlUVView::clientSelect()
{
    const Ray * ray = getIncidentRay();
	Vector2F p(ray->m_origin.x, ray->m_origin.y);
	startTracking(p);
	pickupFeather(p);
	if(m_activeId < 0) return;
	if(interactMode() ==  ToolContext::MoveVertexInUV) {
		m_selectedVert = FeatherLibrary->selectedFeatherExample()->selectVertexInUV(p, m_moveYOnly, m_selectVertWP);
	}
}

bool MlUVView::pickupFeather(const Vector2F & p)
{
	m_activeId = -1;
	if(!FeatherLibrary) return false;
	
	for(MlFeather *f = FeatherLibrary->firstFeatherExample(); FeatherLibrary->hasFeatherExample(); f = FeatherLibrary->nextFeatherExample()) {
		if(!f) continue;
		
		if(f->getBoundingRectangle().isPointInside(p)) {
			m_activeId = f->featherId();
		}
	}
	
	if(m_activeId < 0) return false;
	
	if(!FeatherLibrary->selectFeatherExample(m_activeId))
		std::cout<<"no feather["<<m_activeId<<"]\n";
	
	return true;
}

void MlUVView::clientMouseInput()
{
	if(m_activeId < 0) return;
	const Ray * ray = getIncidentRay();
	Vector2F p(ray->m_origin.x, ray->m_origin.y);
	Vector2F d = updateTracking(p);
	if(interactMode() ==  ToolContext::MoveInUV) {
		FeatherLibrary->selectedFeatherExample()->translateUV(d);
	}
	else if(interactMode() ==  ToolContext::MoveVertexInUV) {
		if(m_selectedVert) {
			if(m_moveYOnly) {
				*m_selectedVert += d.y;
				m_selectVertWP.y += d.y;
			}
			else {
				m_selectedVert[0] += d.x;
				m_selectedVert[1] += d.y;
				m_selectVertWP += d;
			}
			FeatherLibrary->selectedFeatherExample()->computeTexcoord();
		}
	}
}

void MlUVView::drawFeather(MlFeather * f)
{
	BaseDrawer * dr = getDrawer();
	
	Vector3F baseP(f->baseUV());
	
	glPushMatrix();
    Matrix44F s;
	s.setTranslation(baseP);
	
	float * quill = f->getQuilly();
	
	Vector3F a, b;
	
	const unsigned ncv = f->numSegment() + 1;
	
	BezierCurve quillC;
	quillC.createVertices(ncv);
	quillC.m_cvs[0] = s.transform(b);
	
	for(int i = 0; i < f->numSegment(); i++) {
	    b.set(0.f, quill[i], 0.f);
		quillC.m_cvs[i + 1] = s.transform(b);
		s.translate(b);
	}
		
	quillC.computeKnots();
	
	dr->linearCurve(quillC);
	
	dr->smoothCurve(quillC, 2);
	
	
	BezierCurve eRC, eLC, fRC, fLC, gRC, gLC;

	eRC.createVertices(ncv);
	eLC.createVertices(ncv);
	fRC.createVertices(ncv);
	fLC.createVertices(ncv);
	gRC.createVertices(ncv);
	gLC.createVertices(ncv);
	s.setTranslation(baseP);
	
	Vector2F pv;
	for(int i = 0; i <= f->numSegment(); i++) {
		b.set(0.f, quill[i], 0.f);
		Vector2F * vanes = f->getVaneAt(i, 0);
		pv = vanes[0];
	    eRC.m_cvs[i] = s.transform(Vector3F(pv));
		pv += vanes[1];
		fRC.m_cvs[i] = s.transform(Vector3F(pv));
		pv += vanes[2];
		gRC.m_cvs[i] = s.transform(Vector3F(pv));
		vanes = f->getVaneAt(i, 1);
		pv = vanes[0];
	    eLC.m_cvs[i] = s.transform(Vector3F(pv));
		pv += vanes[1];
		fLC.m_cvs[i] = s.transform(Vector3F(pv));
		pv += vanes[2];
		gLC.m_cvs[i] = s.transform(Vector3F(pv));
		s.translate(b);
	}
	eRC.computeKnots();
	fRC.computeKnots();
	gRC.computeKnots();
	eLC.computeKnots();
	fLC.computeKnots();
	gLC.computeKnots();
	
	dr->smoothCurve(gRC, 4);
	dr->smoothCurve(gLC, 4);
	
	const float delta = 1.f / 16.f;
	for(int i=0; i <= 16; i++) {
		float t = delta * i;
		BezierCurve vaneRC;
		vaneRC.createVertices(4);
		vaneRC.m_cvs[0] = quillC.interpolate(t);
		vaneRC.m_cvs[1] = eRC.interpolate(t);
		vaneRC.m_cvs[2] = fRC.interpolate(t);
		vaneRC.m_cvs[3] = gRC.interpolate(t);
		vaneRC.computeKnots();
		dr->smoothCurve(vaneRC, 4);
		
		vaneRC.m_cvs[0] = quillC.interpolate(t);
		vaneRC.m_cvs[1] = eLC.interpolate(t);
		vaneRC.m_cvs[2] = fLC.interpolate(t);
		vaneRC.m_cvs[3] = gLC.interpolate(t);
		vaneRC.computeKnots();
		dr->smoothCurve(vaneRC, 4);
	}

    glPopMatrix();
}

void MlUVView::drawControlVectors(MlFeather * f)
{
	BaseDrawer * dr = getDrawer();
	
	Vector3F baseP(f->baseUV());
	
	glPushMatrix();
    Matrix44F s;
	s.setTranslation(baseP);
	
	float * quill = f->getQuilly();
	Vector3F a, b;
	Vector2F pv;
	for(int i=0; i <= f->numSegment(); i++) {
	    dr->useSpace(s);
		
		const Vector2F * vanes = f->getVaneAt(i, 0);
		pv = vanes[0];
		dr->arrow(Vector3F(0.f, 0.f, 0.f), pv);
		pv += vanes[1];
		dr->arrow(pv - vanes[1], pv);
		pv += vanes[2];
		dr->arrow(pv - vanes[2], pv);
	    
		vanes = f->getVaneAt(i, 1);
		pv = vanes[0];
		dr->arrow(Vector3F(0.f, 0.f, 0.f), pv);
		pv += vanes[1];
		dr->arrow(pv - vanes[1], pv);
		pv += vanes[2];
		dr->arrow(pv - vanes[2], pv);
	    
		if(i < f->numSegment()) {
			b.set(0.f, quill[i], 0.f);
			dr->arrow(Vector3F(0.f, 0.f, 0.f), b);
			s.setTranslation(b);
		}
	}
	glPopMatrix();
}

void MlUVView::drawActiveBound()
{
	if(m_activeId < 0 ) return;
	if(!FeatherLibrary) return;
	
	MlFeather * f = FeatherLibrary->selectedFeatherExample();
	if(!f) return;
	
	getDrawer()->boundingRectangle(f->getBoundingRectangle());
}

void MlUVView::addFeather()
{
	FeatherLibrary->addFeatherExample();
}

void MlUVView::removeSelectedFeather()
{
	if(m_activeId < 0 ) return;
	MlFeather * f = FeatherLibrary->selectedFeatherExample();
	if(!f) return;
	
	QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, tr(" "),
                                    QString("Deleting feather example[%1] will affect its instances, are you sure?").arg(f->featherId()),
                                    QMessageBox::Yes | QMessageBox::Cancel);
    if (reply == QMessageBox::Cancel)
		return;
		
	if(!FeatherLibrary->removeSelectedFeatherExample())
		QMessageBox::information(this, tr(" "),
                                    QString("Cannot delete feather example[%1].").arg(f->featherId()));
}

void MlUVView::changeSelectedFeatherNSegment(int d)
{
	if(m_activeId < 0 ) return;
	if(!FeatherLibrary) return;
	
	MlFeather * f = FeatherLibrary->selectedFeatherExample();
	if(!f) return;
	
	int nseg = f->numSegment() + d;
	if(nseg < 4 || nseg > 32) return;
	
	f->changeNumSegment(d);
}

void MlUVView::chooseImageBackground(std::string & name)
{
	QString selectedFilter;
	QString fileName = QFileDialog::getOpenFileName(this,
							tr("Open .exr image file as the UV texture"),
							tr("info"),
							tr("All Files (*);;EXR Files (*.exr)"),
							&selectedFilter,
							QFileDialog::DontUseNativeDialog);
	if(fileName != "") {
		loadImageBackground(fileName.toUtf8().data());
		name = fileName.toUtf8().data();
	}
}

void MlUVView::loadImageBackground(const std::string & name)
{
	if(!BaseFile::FileExists(name)) return;
	if(!m_image->open(name)) return;
	m_image->verbose();
	makeCurrent();
	m_texId = getDrawer()->loadTexture(m_texId, m_image);
	doneCurrent();
	update();
}
