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
#include "zEXRImage.h"
#include <MlDrawer.h>
#include <BezierCurve.h>
#include <ToolContext.h>
#include <bezierPatch.h>
#include <InverseBilinearInterpolate.h>
#include <MlFeather.h>
#include <MlSkin.h>
BezierPatch testbez;
BezierPatch testsplt[4];
InverseBilinearInterpolate invbil;
PatchSplitContext childUV[4];
MlFeather fea;
BaseDrawer * dr;

void createFeather()
{
    fea.createNumSegment(4);
    float * quill = fea.quilly();
    quill[0] = 4.f;
    quill[1] = 2.7f;
    quill[2] = 1.7f;
    quill[3] = .9f;
    
    Vector2F * vanes = fea.vaneAt(0, 0);
    vanes[0].set(1.f, 1.5f);
    vanes[1].set(2.f, 1.5f);
    vanes[2].set(3.f, 2.7f);
    vanes = fea.vaneAt(0, 1);
    vanes[0].set(-1.f, .5f);
    vanes[1].set(-1.6f, 1.2f);
    vanes[2].set(-2.4f, 2.2f);
    
    vanes = fea.vaneAt(1, 0);
    vanes[0].set(1.2f, 1.1f);
    vanes[1].set(1.8f, 1.1f);
    vanes[2].set(3.f, 2.2f);
    vanes = fea.vaneAt(1, 1);
    vanes[0].set(-1.2f, .62f);
    vanes[1].set(-1.7f, 1.2f);
    vanes[2].set(-2.4f, 2.2f);
    
    vanes = fea.vaneAt(2, 0);
    vanes[0].set(.6f, .25f);
    vanes[1].set(1.f, .6f);
    vanes[2].set(1.6f, 1.1f);
    vanes = fea.vaneAt(2, 1);
    vanes[0].set(-.7f, .35f);
    vanes[1].set(-1.f, .5f);
    vanes[2].set(-1.5f, 1.1f);
    
    vanes = fea.vaneAt(3, 0);
    vanes[0].set(.4f, .3f);
    vanes[1].set(.5f, .5f);
    vanes[2].set(.7f, .6f);
    vanes = fea.vaneAt(3, 1);
    vanes[0].set(-.4f, .4f);
    vanes[1].set(-.5f, .6f);
    vanes[2].set(-.6f, .7f);
    
    vanes = fea.vaneAt(4, 0);
    vanes[0].set(.0f, .1f);
    vanes[1].set(.01f, .2f);
    vanes[2].set(.01f, .3f);
    vanes = fea.vaneAt(4, 1);
    vanes[0].set(-.01f, .1f);
    vanes[1].set(-.02f, .2f);
    vanes[2].set(-.02f, .3f);
}

void drawFeather()
{
    glPushMatrix();
    Matrix44F s;
	s.setTranslation(5.f, 3.f, 4.f);
	
	float * quill = fea.getQuilly();
	
	Vector3F a, b;
	
	BezierCurve quillC;
	quillC.createVertices(5);
	quillC.m_cvs[0] = s.transform(b);
	
	for(int i = 0; i < 4; i++) {
	    b.set(0.f, quill[i], 0.f);
		quillC.m_cvs[i + 1] = s.transform(b);
		s.translate(b);
	}
		
	quillC.computeKnots();
	
	dr->linearCurve(quillC);
	
	dr->smoothCurve(quillC, 8);
	
	
	BezierCurve eRC, eLC, fRC, fLC, gRC, gLC;

	eRC.createVertices(5);
	eLC.createVertices(5);
	fRC.createVertices(5);
	fLC.createVertices(5);
	gRC.createVertices(5);
	gLC.createVertices(5);
	s.setTranslation(5.f, 3.f, 4.f);
	
	for(int i = 0; i < 5; i++) {
		b.set(0.f, quill[i], 0.f);
		Vector2F * vanes = fea.getVaneAt(i, 0);
	    eRC.m_cvs[i] = s.transform(Vector3F(vanes[0]));
		fRC.m_cvs[i] = s.transform(Vector3F(vanes[1]));
		gRC.m_cvs[i] = s.transform(Vector3F(vanes[2]));
		vanes = fea.getVaneAt(i, 1);
	    eLC.m_cvs[i] = s.transform(Vector3F(vanes[0]));
		fLC.m_cvs[i] = s.transform(Vector3F(vanes[1]));
		gLC.m_cvs[i] = s.transform(Vector3F(vanes[2]));
		s.translate(b);
	}
	eRC.computeKnots();
	fRC.computeKnots();
	gRC.computeKnots();
	eLC.computeKnots();
	fLC.computeKnots();
	gLC.computeKnots();
	
	const float delta = 1.f / 52.f;
	for(int i=0; i <= 52; i++) {
		float t = delta * i;
		BezierCurve vaneRC;
		vaneRC.createVertices(4);
		vaneRC.m_cvs[0] = quillC.interpolate(t);
		vaneRC.m_cvs[1] = eRC.interpolate(t);
		vaneRC.m_cvs[2] = fRC.interpolate(t);
		vaneRC.m_cvs[3] = gRC.interpolate(t);
		vaneRC.computeKnots();
		dr->smoothCurve(vaneRC, 2);
		
		vaneRC.m_cvs[0] = quillC.interpolate(t);
		vaneRC.m_cvs[1] = eLC.interpolate(t);
		vaneRC.m_cvs[2] = fLC.interpolate(t);
		vaneRC.m_cvs[3] = gLC.interpolate(t);
		vaneRC.computeKnots();
		dr->smoothCurve(vaneRC, 2);
	}
	
	s.setTranslation(5.f, 3.f, 4.f);
	for(int i=0; i <= 4; i++) {
	    b.set(0.f, quill[i], 0.f);
	    
	    dr->useSpace(s);
		dr->arrow(Vector3F(0.f, 0.f, 0.f), b);
		
		Vector2F * vanes = fea.getVaneAt(i, 0);
		dr->arrow(Vector3F(0.f, 0.f, 0.f), vanes[0]);
		dr->arrow(vanes[0], vanes[1]);
		dr->arrow(vanes[1], vanes[2]);
	    
		vanes = fea.getVaneAt(i, 1);
		dr->arrow(Vector3F(0.f, 0.f, 0.f), vanes[0]);
		dr->arrow(vanes[0], vanes[1]);
		dr->arrow(vanes[1], vanes[2]);
	    
	    s.setTranslation(b);
	}
	
    glPopMatrix();
}

GLWidget::GLWidget(QWidget *parent) : SingleModelView(parent)
{
    dr = getDrawer();
    createFeather();
    testbez.evaluateContolPoints();testbez.decasteljauSplit(testsplt);
invbil.setVertices(Vector3F(-1,0,0), Vector3F(1,0,0), Vector3F(-1,1,0), Vector3F(2,2,0));
Vector3F P(1.51f,0.71f,0.f);
Vector2F testuv = invbil(P);
printf("invbilinear %f %f\n", testuv.x, testuv.y);

//Vector2F uv(0.005f, 0.00485f);
//testuv = invbil.evalBiLinear(uv);
//printf("bilinear %f %f\n", testuv.x, testuv.y);


	m_accmesh = new AccPatchMesh;
#ifdef WIN32
	std::string filename("D:/aphid/mdl/torus.m");
#else
	std::string filename("/Users/jianzhang/aphid/mdl/torus.m");
#endif

	loadMesh(filename);
	
#ifdef WIN32
	_image = new ZEXRImage("D:/aphid/catmullclark/disp.exr");
#else
	_image = new ZEXRImage("/Users/jianzhang/aphid/catmullclark/disp.exr");
#endif
	//if(_image->isValid()) qDebug()<<"image is loaded";
	// use displacement map inside bezier drawer
	//_tess->setDisplacementMap(_image);

	m_skin = new MlSkin;
	m_skin->setBodyMesh(m_accmesh, m_topo);
	
	m_featherDrawer = new MlDrawer;
	m_featherDrawer->updateMesh(m_accmesh);
	
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
	m_featherDrawer->drawAcc();
	//getDrawer()->drawKdTree(getTree());
	m_featherDrawer->drawFeather(m_skin);
	
	glPushMatrix();
	
	Matrix44F s;
	//s.setTranslation(5.f, 3.f, 4.f);
	s.rotateX(1.1f);
	s.rotateY(0.97f);
	getDrawer()->useSpace(s);
	getDrawer()->coordsys(15.f);
	
	Matrix44F b;
	b.rotateY(0.97f);
	getDrawer()->useSpace(b);
	//getDrawer()->coordsys(10.f);
	glPopMatrix();
	
	
	//glPushMatrix();
	
	
	Matrix33F c;
	c.rotateX(6.1f);
	c.rotateY(0.97f);
	//c.rotateZ(-0.67f);
	//c.setTranslation(0.f, 0.f, 14.f);
	
	//
	
	//Matrix44F t;
	//t.rotateZ(-0.67f);
	//t.setTranslation(5.f, 8.f, 14.f);
	
	//c.multiply(t);
	Vector3F p(4.f, 3.f, 5.f);
	
	//getDrawer()->useSpace(c);
	getDrawer()->coordsys(c, 5.f, &p);
	
	//glPopMatrix();
	
	//drawFeather();
	//drawSelection();
	showBrush();
}

void GLWidget::setSelectionAsWale(int bothSide)
{
}

void GLWidget::changeWaleResolution(int change)
{
}

void GLWidget::changeCourseResolution(int change)
{
}

void GLWidget::loadMesh(std::string filename)
{
	ESMUtil::ImportPatch(filename.c_str(), mesh());
	buildTopology();
	m_accmesh->setup(m_topo);
	buildTree();
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
		addFeather();
	}
}

void GLWidget::clientMouseInput()
{
    Vector3F hit;
	Ray ray = *getIncidentRay();
	if(interactMode() == ToolContext::SelectVertex) {
		pickupComponent(ray, hit);
	}
	else if(interactMode() == ToolContext::CreateBodyContourFeather) {
		brush()->setToeByIntersectNormal(&ray);
		m_skin->growFeather(brush()->toeDisplacement());
	}
}

void GLWidget::clientDeselect()
{
    if(interactMode() == ToolContext::CreateBodyContourFeather) {
		m_skin->finishCreateFeather();
	}
}

PatchMesh * GLWidget::mesh() const
{
	return m_accmesh;
}

void GLWidget::addFeather()
{
	IntersectionContext * ctx = getIntersectionContext();
    if(!ctx->m_success) return;
	
	brush()->setSpace(ctx->m_hitP, ctx->m_hitN);
	brush()->resetToe();
	
	Vector3F rr = getIncidentRay()->m_dir;
	rr.reverse();
	if(rr.dot(brush()->normal()) < .34f) return;
	
	const unsigned iface = ctx->m_componentIdx;
	for(unsigned i = 0; i < brush()->getNumDarts(); i++) {
		Ray r = brush()->getObjectRay(i);
		ctx->reset(r);
		if(!m_accmesh->intersect(iface, ctx)) {
			Vector3F hit;
			hitTest(r, hit);
		}
		if(ctx->m_success && ctx->m_hitN.dot(brush()->normal()) > .8f) {
			MlCalamus c;
			c.bindToFace(ctx->m_componentIdx, ctx->m_patchUV.x, ctx->m_patchUV.y);
			c.setRotateY(brush()->getPitch());
			m_skin->createFeather(c, ctx->m_hitP, brush()->minDartDistance());
		}
	}
}
//:~
