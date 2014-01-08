/*
 *  BaseDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif

#include "BaseDrawer.h"
#include "Matrix33F.h"
#include <cmath>
#include <zEXRImage.h>
#include <BaseTransform.h>
#include <TransformManipulator.h>
#include <SkeletonJoint.h>
#include <LineBuffer.h>
#include <AdaptableStripeBuffer.h>

BaseDrawer::BaseDrawer () : m_wired(0) 
{
	m_sphere = new GeodesicSphereMesh(8);
	m_pyramid = new PyramidMesh;
	m_circle = new CircleCurve;
	m_cube = new CubeMesh;
	m_activeColor.set(0.f, .8f, .2f);
	m_inertColor.set(0.1f, 0.6f, 0.1f);
}

BaseDrawer::~BaseDrawer () 
{
	delete m_sphere;
	delete m_pyramid;
	delete m_circle;
	delete m_cube;
}

void BaseDrawer::initializeProfile()
{
	m_markerProfile = GProfile(false, true, false, false, false);
	m_surfaceProfile = GProfile(true, true, false, false, true);
	surfaceMat = new GMaterial(Color4(0.1, 0.1, 0.1, 1.0),Color4(0.8, 0.8, 0.8, 1.0),Color4(0.4, 0.4, 0.3, 1.0),Color4(0.0, 0.0, 0.0, 1.0), 64.f);
	m_surfaceProfile.m_material = surfaceMat;
	m_wireProfile = GProfile(false, true, true, false, false);
	majorLit.activate();
	fillLit.m_LightID = GL_LIGHT1;
	fillLit.m_Position = Float4(0.f, 0.f, -1000.f, 1.f);
	fillLit.activate();
}

void BaseDrawer::cube(const Vector3F & p, const float & size) const
{
	glPushMatrix();

	glTranslatef(p.x, p.y, p.z);
	glScalef(size, size, size);
	
	drawMesh(m_cube);
	glPopMatrix();	
}

void BaseDrawer::box(float width, float height, float depth)
{
	glBegin(GL_LINES);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(width, 0.f, 0.f);
	
	glColor3f(0.f, 1.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, height, 0.f);
	
	glColor3f(0.f, 0.f, 1.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, depth);
	
	glColor3f(0.23f, 0.23f, 0.24f);
	
	glVertex3f(width, 0.f, 0.f);
	glVertex3f(width, 0.f, depth);
	
	glVertex3f(width, 0.f, depth);
	glVertex3f(0.f, 0.f, depth);
	
	glVertex3f(0.f, height, 0.f);
	glVertex3f(width, height, 0.f);
	
	glVertex3f(width, height, 0.f);
	glVertex3f(width, height, depth);
	
	glVertex3f(width, height, depth);
	glVertex3f(0.f, height, depth);
	
	glVertex3f(0.f, height, depth);
	glVertex3f(0.f, height, 0.f);
	
	glVertex3f(width, 0.f, 0.f);
	glVertex3f(width, height, 0.f);
	
	glVertex3f(width, 0.f, depth);
	glVertex3f(width, height, depth);
	
	glVertex3f(0.f, 0.f, depth);
	glVertex3f(0.f, height, depth);
	glEnd();
}

void BaseDrawer::solidCube(float x, float y, float z, float size)
{
	setWired(0);
	glPushMatrix();

	//const float hsize = size * 0.5f; 
	//glTranslatef(x - hsize, y - hsize, z - hsize);
	glTranslatef(x, y, z);
	glScalef(size, size, size);
	
	drawMesh(m_cube);
	glPopMatrix();	
}

void BaseDrawer::transform(BaseTransform * t)
{
	Matrix44F ws = t->worldSpace();
	glPushMatrix();
	useSpace(ws);
	glBegin(GL_LINES);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(8.f, 0.f, 0.f);
	glColor3f(0.f, 1.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 8.f, 0.f);
	glColor3f(0.f, 0.f, 1.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 8.f);
	glEnd();
	glPopMatrix();
	
	Matrix44F ps;
	t->parentSpace(ps);
	glPushMatrix();
	useSpace(ps);
	
	const Vector3F p = t->translation();
	glPushMatrix();
	
	glTranslatef(p.x, p.y, p.z);
	spinPlanes(t);
		
	glPopMatrix();
}

void BaseDrawer::manipulator(TransformManipulator * m)
{
	if(m->isDetached()) return;
	
	Matrix44F ps;
	m->parentSpace(ps);
	glPushMatrix();
	useSpace(ps);

	const Vector3F p = m->translation();
	glPushMatrix();
	glTranslatef(p.x, p.y, p.z);

	if(m->mode() == ToolContext::MoveTransform) {
		moveHandle(m->rotateAxis(), 1);
		spinHandle(m, 0);
	}
	else
		spinHandle(m, 1);
		
	glPopMatrix();
	glPopMatrix();
	
	if(!m->started()) return;
	
	setGrey(.5f);
	if(m->mode() == ToolContext::MoveTransform)
		arrow(m->startPoint(), m->currentPoint());
	else {
		arrow(m->worldSpace().getTranslation(), m->startPoint());
		arrow(m->worldSpace().getTranslation(), m->currentPoint());
	}
}

void BaseDrawer::skeletonJoint(SkeletonJoint * joint)
{
	transform(joint);
	if(joint->numChildren() < 1) return;
	
	Matrix44F ws = joint->worldSpace();
	glPushMatrix();
	useSpace(ws);
	setGrey(.67f);
	m_wireProfile.apply();
	Vector3F to, fm;
	float d;
	for(unsigned i = 0; i < joint->numChildren(); i++) {
		to = joint->child(i)->translation();
		fm = to.normal() * 1.f;
		d = to.length();
		to.normalize();
		to *= d - 1.f;
		arrow2(fm, to, 4.f);
	}
	glPopMatrix();
}

void BaseDrawer::moveHandle(int axis, bool active)
{
	if(axis != BaseTransform::AX && active)
		glColor3f(1.f, 0.f, 0.f);
	else
		setGrey(.5f);
		
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(8.f, 0.f, 0.f));
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(-8.f, 0.f, 0.f));
	
	if(axis != BaseTransform::AY && active)
		glColor3f(0.f, 1.f, 0.f);
	else
		setGrey(.5f);
		
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, 8.f, 0.f));
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, -8.f, 0.f));
	
	if(axis != BaseTransform::AZ && active)
		glColor3f(0.f, 0.f, 1.f);
	else
		setGrey(.5f);
		
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, 0.f, 8.f));
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, 0.f, -8.f));
}

void BaseDrawer::spinPlanes(BaseTransform * t)
{
	m_circle->setRadius(8.f);
	glPushMatrix();

	Matrix44F mat;
	Vector3F a;
	glPushMatrix();
	a = t->rotatePlane(TransformManipulator::AZ);
	mat.setFrontOrientation(a);
	useSpace(mat);
	setColor(0.f, 0.f, 1.f);
	linearCurve(*m_circle);
	glPopMatrix();
	
	glPushMatrix();
	a = t->rotatePlane(TransformManipulator::AY);
	mat.setFrontOrientation(a);
	useSpace(mat);
	setColor(0.f, 1.f, 0.f);
	linearCurve(*m_circle);
	glPopMatrix();
	
	glPushMatrix();
	a = t->rotatePlane(TransformManipulator::AX);
	mat.setFrontOrientation(a);
	useSpace(mat);
	setColor(1.f, 0.f, 0.f);
	linearCurve(*m_circle);
	glPopMatrix();
	glPopMatrix();
}

void BaseDrawer::spinHandle(TransformManipulator * m, bool active)
{
	m_circle->setRadius(8.f);
	glPushMatrix();

	Matrix44F mat;
	Vector3F a;
	glPushMatrix();
	a = m->rotatePlane(TransformManipulator::AZ);
	mat.setFrontOrientation(a);
	useSpace(mat);
	if(m->rotateAxis() == TransformManipulator::AZ && active) setColor(0.f, 0.f, 1.f);
	else setGrey(.5f);
	linearCurve(*m_circle);
	glPopMatrix();
	
	glPushMatrix();
	a = m->rotatePlane(TransformManipulator::AY);
	mat.setFrontOrientation(a);
	useSpace(mat);
	if(m->rotateAxis() == TransformManipulator::AY && active) setColor(0.f, 1.f, 0.f);
	else setGrey(.5f);
	linearCurve(*m_circle);
	glPopMatrix();
	
	glPushMatrix();
	a = m->rotatePlane(TransformManipulator::AX);
	mat.setFrontOrientation(a);
	useSpace(mat);
	if(m->rotateAxis() == TransformManipulator::AX && active) setColor(1.f, 0.f, 0.f);
	else setGrey(.5f);
	linearCurve(*m_circle);
	glPopMatrix();
	glPopMatrix();
}

void BaseDrawer::setGrey(float g)
{
    glColor3f(g, g, g);
}

void BaseDrawer::setColor(float r, float g, float b) const
{
	glColor3f(r, g, b);
}

void BaseDrawer::end() const
{
    if(m_wired) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnd();
}

void BaseDrawer::beginSolidTriangle()
{
	glBegin(GL_TRIANGLES);
}

void BaseDrawer::beginWireTriangle()
{
    m_wired = 1;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
}

void BaseDrawer::beginLine()
{
	glBegin(GL_LINES);
}

void BaseDrawer::beginPoint(float x) const
{
	glPointSize(x);
	glBegin(GL_POINTS);
}

void BaseDrawer::beginQuad()
{
	glBegin(GL_QUADS);
}

void BaseDrawer::aVertex(float x, float y, float z)
{
	glVertex3f(x, y, z);
}

void BaseDrawer::drawSphere()
{
	const float angleDelta = 3.14159269f / 36.f;
	float a0, a1, b0, b1;
	glBegin(GL_LINES);
	for(int i=0; i<72; i++) {
		float angleMin = angleDelta * i;
		float angleMax = angleMin + angleDelta;
		
		a0 = cos(angleMin);
		b0 = sin(angleMin);
		
		a1 = cos(angleMax);
		b1 = sin(angleMax);
		
		glVertex3f(a0, b0, 0.f);
		glVertex3f(a1, b1, 0.f);
		
		glVertex3f(a0, 0.f, b0);
		glVertex3f(a1, 0.f, b1);
		
		glVertex3f(0.f, a0, b0);
		glVertex3f(0.f, a1, b1);
	}
	glEnd();
}

void BaseDrawer::drawCircleAround(const Vector3F& center)
{
	Vector3F nor(center.x, center.y, center.z);
	Vector3F tangent = nor.perpendicular();
	
	Vector3F v0 = tangent * 0.1f;
	Vector3F p;
	const float delta = 3.14159269f / 9.f;
	
	glBegin(GL_LINES);
	for(int i = 0; i < 18; i++) {
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
		
		v0.rotateAroundAxis(nor, delta);
		
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
}

void BaseDrawer::quadMesh(const BaseMesh * mesh) const
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	
	glDrawElements(GL_QUADS, mesh->getNumPolygonFaceVertices(), GL_UNSIGNED_INT, mesh->getPolygonIndices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::drawMesh(const BaseMesh * mesh, const BaseDeformer * deformer) const
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

	glDrawElements(GL_TRIANGLES, mesh->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::drawPolygons(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	Vector3F * p = mesh->getVertices();
	if(deformer) p = deformer->getDeformedP();
	
	const unsigned nf = mesh->getNumPolygons();
	unsigned fi = 0;
	unsigned *fc = mesh->getPolygonCounts();
	unsigned *fv = mesh->getPolygonIndices();
	for(unsigned i = 0; i < nf; i++) {
		glBegin(GL_POLYGON);
		for(unsigned j =0; j < fc[i]; j++) {
			vertex(p[fv[fi + j]]);
		}
		glEnd();
		fi += fc[i];
	}
}

void BaseDrawer::drawPoints(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

	glDrawArrays(GL_POINTS, 0, mesh->getNumVertices());

	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::showNormal(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getNormals());
	
	glDrawElements(GL_TRIANGLES, mesh->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::edge(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	if(mesh->m_numQuadVertices < 4) return;

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedP());

// draw a cube
	glDrawElements(GL_QUADS, mesh->m_numQuadVertices, GL_UNSIGNED_INT, mesh->m_quadIndices);

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::field(const BaseField * f)
{
	BaseMesh *mesh = f->m_mesh;
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, (GLfloat*)f->getColor());
	
// draw a cube
	glDrawElements(GL_TRIANGLES, mesh->getNumTriangleFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::tangentFrame(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	unsigned nv = mesh->getNumVertices();
	Vector3F * v = mesh->getVertices();
	if(deformer)
		v = deformer->getDeformedP();
		
	float m[16];
	for(unsigned i = 0; i < nv; i++) {
		glPushMatrix();
		//glTranslatef(v[i].x, v[i].y, v[i].z);
		Matrix33F orient = mesh->getTangentFrame(i);
    
    m[0] = orient(0, 0); m[1] = orient(0, 1); m[2] = orient(0, 2); m[3] = 0.0;
    m[4] = orient(1, 0); m[5] = orient(1, 1); m[6] = orient(1, 2); m[7] = 0.0;
    m[8] = orient(2, 0); m[9] = orient(2, 1); m[10] = orient(2, 2); m[11] = 0.0;
    m[12] = v[i].x;
	m[13] = v[i].y;
	m[14] = v[i].z; 
	m[15] = 1.0;
    glMultMatrixf((const GLfloat*)m);
		coordsys();
				glPopMatrix();
	}
}

void BaseDrawer::boundingRectangle(const BoundingRectangle & b) const
{
	glBegin(GL_LINE_LOOP);
	glVertex3f(b.getMin(0), b.getMin(1), 0.f);
	glVertex3f(b.getMax(0), b.getMin(1), 0.f);
	glVertex3f(b.getMax(0), b.getMax(1), 0.f);
	glVertex3f(b.getMin(0), b.getMax(1), 0.f);
	glEnd();
}

void BaseDrawer::boundingBox(const BoundingBox & b) const
{
	Vector3F corner0(b.getMin(0), b.getMin(1), b.getMin(2));
	Vector3F corner1(b.getMax(0), b.getMax(1), b.getMax(2));
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glEnd();
}

void BaseDrawer::triangle(const BaseMesh * mesh, unsigned idx)
{
	beginSolidTriangle();
	Vector3F *v = mesh->getVertices();
	unsigned *i = mesh->getIndices();
	
	Vector3F & a = v[i[idx * 3]];
	Vector3F & b = v[i[idx * 3 + 2]];
	Vector3F & c = v[i[idx * 3 + 1]];
	
	glVertex3f(a.x, a.y, a.z);
	glVertex3f(b.x, b.y, b.z);
	glVertex3f(c.x, c.y, c.z);
	end();
}

void BaseDrawer::patch(const BaseMesh * mesh, unsigned idx)
{
	beginQuad();
	Vector3F *v = mesh->getVertices();
	unsigned *i = mesh->getQuadIndices();
	
	Vector3F & a = v[i[idx * 4]];
	Vector3F & b = v[i[idx * 4 + 1]];
	Vector3F & c = v[i[idx * 4 + 2]];
	Vector3F & d = v[i[idx * 4 + 3]];
	
	glVertex3f(a.x, a.y, a.z);
	glVertex3f(b.x, b.y, b.z);
	glVertex3f(c.x, c.y, c.z);
	glVertex3f(d.x, d.y, d.z);
	end();
}

void BaseDrawer::components(SelectionArray * arr)
{
    BaseMesh *mesh = (BaseMesh *)arr->getGeometry();
    if(arr->getComponentFilterType() == PrimitiveFilter::TFace) {
        const unsigned numFace = arr->numFaces();
        for(unsigned i = 0; i < numFace; i++) {
            triangle((const BaseMesh *)mesh, arr->getFaceId(i));
        }
    }
    else if(arr->getComponentFilterType() == PrimitiveFilter::TVertex) {
        const unsigned numVert = arr->numVertices();
		if(numVert < 1) return;
		BaseCurve curve;
		glDisable(GL_DEPTH_TEST);
		for(unsigned i = 0; i < numVert; i++) {
			Vector3F p = arr->getVertexP(i);
			solidCube(p.x, p.y, p.z, 0.2f);
			curve.addVertex(p);
		}
		glEnable(GL_DEPTH_TEST);
		
		if(arr->hasVertexPath()) {
		    curve.finishAddVertex();
			curve.computeKnots();
			glLineWidth(2.f);
			linearCurve(curve);
			glLineWidth(1.f);
		}
    }
}

void BaseDrawer::primitive(Primitive * prim)
{
	BaseMesh *geo = (BaseMesh *)prim->getGeometry();//printf("prim %i ", geo->entityType());
	const unsigned iface = prim->getComponentIndex();
	if(geo->isTriangleMesh())
		triangle((const BaseMesh *)geo, iface);
	else {
		patch((const BaseMesh *)geo, iface);
		}
}

void BaseDrawer::coordsys(float scale) const
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor3f(1.f, 0.f, 0.f);
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(scale, 0.f, 0.f));
	glColor3f(0.f, 1.f, 0.f);
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, scale, 0.f));
	glColor3f(0.f, 0.f, 1.f);
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, 0.f, scale));
}

void BaseDrawer::arrow0(const Vector3F & at, const Vector3F & dir, float l, float w) const
{
	Matrix44F space;
	space.setFrontOrientation(dir.normal());
	space.setTranslation(at);
	glPushMatrix();
	useSpace(space);
	glRotatef(90.f, 1.f, 0.f, 0.f);
	glScalef(w, l, w);
	drawMesh(m_pyramid);
	glPopMatrix();
}

void BaseDrawer::arrow(const Vector3F& origin, const Vector3F& dest) const
{
	glBegin( GL_LINES );
	glVertex3f(origin.x, origin.y, origin.z);
	glVertex3f(dest.x, dest.y, dest.z); 
	glEnd();
	
	const Vector3F r = dest - origin;
	const float arrowLength = r.length() * 0.13f;
	const float arrowWidth = arrowLength * .31f;
	arrow0(dest, r, arrowLength, arrowWidth);
}

void BaseDrawer::arrow2(const Vector3F& origin, const Vector3F& dest, float width) const
{	
	Vector3F r = dest - origin;
	arrow0(dest, r, r.length() * .8f, width);
	
	r = origin - dest;
	arrow0(origin, r, r.length() * .2f, width);
}

void BaseDrawer::coordsys(const Matrix33F & orient, float scale, Vector3F * p) const
{
	glPushMatrix();
	if(p) glTranslatef(p->x, p->y, p->z);
	float m[16];
	orient.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
	coordsys(scale);
	
	glPopMatrix();
}

void BaseDrawer::useSpace(const Matrix44F & s) const
{
	float m[16];
	s.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
}

void BaseDrawer::useSpace(const Matrix33F & s) const
{
	float m[16];
	s.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
}

void BaseDrawer::setWired(char var)
{
	if(var) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void BaseDrawer::setCullFace(char var)
{
	if(var) glEnable(GL_CULL_FACE);
	else glDisable(GL_CULL_FACE);
}

void BaseDrawer::anchor(Anchor *a, char active)
{
	glPushMatrix();
	float m[16];
    
    a->spaceMatrix(m);
    
    glMultMatrixf((const GLfloat*)m);
	glDisable(GL_DEPTH_TEST);
	
	unsigned nouse;
	Anchor::AnchorPoint * ap;
	for(ap = a->firstPoint(nouse); a->hasPoint(); ap = a->nextPoint(nouse)) {
		if(!active) colorAsInert();
		else colorAsActive();
	
		Vector3F p = ap->p;
		glPushMatrix();
		glTranslatef(p.x, p.y, p.z);
		glScalef(0.2f, 0.2f, 0.2f);
		glRotatef(180.f, 1.0f, .0f, 0.0f);
		drawMesh(m_pyramid);
		glPopMatrix();
	}
	glEnable(GL_DEPTH_TEST);
	
	if(a->numPoints() > 1) {
		BaseCurve curve;
		for(unsigned i = 0; i < a->numPoints(); i++)
			curve.addVertex(a->getPoint(i)->p);
		curve.finishAddVertex();
		curve.computeKnots();
		linearCurve(curve);
	}
	glPopMatrix();
}

void BaseDrawer::spaceHandle(SpaceHandle * hand)
{
	if(!hand) return;
	glPushMatrix();
	float m[16];
    
    hand->spaceMatrix(m);
	glMultMatrixf((const GLfloat*)m);
	glDisable(GL_DEPTH_TEST);
	
	colorAsActive();
	sphere(hand->getSize());
	
	glEnable(GL_DEPTH_TEST);
	glPopMatrix();
}

void BaseDrawer::sphere(float size) const
{
	glPushMatrix();
	glScalef(size, size, size);
	drawMesh(m_sphere);
	glPopMatrix();
}

void BaseDrawer::linearCurve(const BaseCurve & curve)
{
    glDisable(GL_DEPTH_TEST);
	float t;
	Vector3F p;
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i < curve.numVertices(); i++) {
		p = curve.getCv(i);
		t = curve.getKnot(i);
		//setColor(1.f - t, 0.f, t);
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
	glEnable(GL_DEPTH_TEST);
}

void BaseDrawer::smoothCurve(BaseCurve & curve, short deg)
{
	glDisable(GL_DEPTH_TEST);
	float t;
	const unsigned nseg = (curve.numVertices() - 1) * deg;
	const float delta = 1.f / nseg;
	Vector3F p;
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i <= nseg; i++) {
		t = delta * i;
		p = curve.interpolate(t);
		setColor(1.f - t, 0.f, t);
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
	glEnable(GL_DEPTH_TEST);
}

void BaseDrawer::hiddenLine(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	setCullFace(1);
	setWired(0);
	setGrey(0.4f);
	drawMesh(mesh, deformer);
	setWired(1);
	setGrey(0.9f);
	edge(mesh, deformer);
	setCullFace(0);
}

void BaseDrawer::colorAsActive()
{
	glColor3f(m_activeColor.x, m_activeColor.y, m_activeColor.z);
}

void BaseDrawer::colorAsInert()
{
	glColor3f(m_inertColor.x, m_inertColor.y, m_inertColor.z);
}

void BaseDrawer::vertex(const Vector3F & v) const
{
	glVertex3f(v.x, v.y, v.z);
}

void BaseDrawer::vertexWithOffset(const Vector3F & v, const Vector3F & o)
{
	glVertex3f(v.x, v.y, v.z);
	glVertex3f(v.x + o.x, v.y + o.y, v.z + o.z);
}

void BaseDrawer::circleAt(const Vector3F & pos, const Vector3F & nor)
{
    glPushMatrix();
    Matrix44F mat;
    mat.setTranslation(pos);
    mat.setFrontOrientation(nor);
	useSpace(mat);
    linearCurve(*m_circle);
    glPopMatrix();
}

void BaseDrawer::circleAt(const Matrix44F & mat, float radius)
{
    glPushMatrix();
    useSpace(mat);
	m_circle->setRadius(radius);
    linearCurve(*m_circle);
    glPopMatrix();
}

void BaseDrawer::useDepthTest(char on) const
{
	if(on) glEnable(GL_DEPTH_TEST);
	else glDisable(GL_DEPTH_TEST);
}

int BaseDrawer::addTexture()
{
	GLuint tex = 0;
	m_textureNames.push_back(tex);
	return m_textureNames.size() - 1;
}

void BaseDrawer::clearTexture(int idx)
{
	if(idx < 0) return;
	if(idx + 1 > m_textureNames.size()) return;
	GLuint * tex = &m_textureNames[idx];
	if(*tex > 0) glDeleteTextures(1, tex);
}

int BaseDrawer::loadTexture(int idx, ZEXRImage * image)
{
	if(idx < 0) idx = addTexture();
	
	clearTexture(idx);
	
	GLuint * tex = &m_textureNames[idx];
	
	glGenTextures(1, tex);
	
	glBindTexture(GL_TEXTURE_2D, *tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE); 
	const int w = image->_mipmaps[0]->getWidth();
	if(image->m_channelRank == BaseImage::RGB)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, w, w, 0, GL_RGB, GL_HALF_FLOAT_ARB, image->_mipmaps[0]->getPixels());
	else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, w, w, 0, GL_RGBA, GL_HALF_FLOAT_ARB, image->_mipmaps[0]->getPixels());

	glBindTexture( GL_TEXTURE_2D, 0 );
	
	return idx;
}

void BaseDrawer::texture(int idx)
{	
	glDisable(GL_LIGHTING);
	glColor3f(1, 1, 1);
	glEnable(GL_TEXTURE_2D);
	
	bindTexture(idx);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex3f(0, 0, 0);
	glTexCoord2f(1, 0);
	glVertex3f(1, 0, 0);
	glTexCoord2f(1, 1);
	glVertex3f(1, 1, 0);
	glTexCoord2f(0, 1);
	glVertex3f(0, 1, 0);
	glEnd();
	unbindTexture();
	glDisable(GL_TEXTURE_2D);
}

void BaseDrawer::bindTexture(int idx)
{
	glBindTexture(GL_TEXTURE_2D, m_textureNames[idx]);
}

void BaseDrawer::unbindTexture()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

void BaseDrawer::drawLineBuffer(LineBuffer * line) const
{
    if(line->numVertices() < 2) return;
    glBegin(GL_LINES);
    for(unsigned i = 0; i < line->numVertices(); i++)
    {
        Vector3F p = line->vertices()[i];
        glVertex3f(p.x , p.y, p.z);
    }
    glEnd();
}

void BaseDrawer::perVertexVector(BaseMesh * mesh, const std::string & name)
{
	Vector3F *pvv = mesh->perVertexVector(name);
	if(!pvv) return;
	Vector3F * p = mesh->getVertices();
	Vector3F q;
	const unsigned nv = mesh->getNumVertices();
	glBegin(GL_LINES);
	for(unsigned i = 0; i < nv; i++) {
		q = p[i];
		glVertex3fv((float *)&q);
		q += pvv[i];
		glVertex3fv((float *)&q);
	}
	glEnd();
}

void BaseDrawer::vertexNormal(BaseMesh * mesh)
{
	Vector3F *nor = mesh->getNormals();
	Vector3F * p = mesh->getVertices();
	const unsigned nv = mesh->getNumVertices();
	Vector3F q;
	glBegin(GL_LINES);
	for(unsigned i = 0; i < nv; i++) {
		q = p[i];
		glVertex3fv((float *)&q);
		q += nor[i];
		glVertex3fv((float *)&q);
	}
	glEnd();
}

void BaseDrawer::lines(const std::vector<Vector3F> & vs)
{
	glBegin(GL_LINES);
	std::vector<Vector3F>::const_iterator it = vs.begin();
	for(; it != vs.end(); ++it)
		glVertex3fv((float *)&(*it));
	glEnd();
}

void BaseDrawer::lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs) const
{
	unsigned acc = 0;
	for(unsigned i = 0; i < num; i++) {
		glBegin(GL_LINE_STRIP);
		for(unsigned j = 0; j < nv[i]; j++) {
			glVertex3fv((float *)&vs[acc]);
			acc++;
		}
		glEnd();
	}
}

void BaseDrawer::lineStripes(const unsigned & num, unsigned * nv, Vector3F * vs, Vector3F * cs) const
{
	unsigned acc = 0;
	for(unsigned i = 0; i < num; i++) {
		glBegin(GL_LINE_STRIP);
		for(unsigned j = 0; j < nv[i]; j++) {
			glColor3fv((float *)&cs[acc]);
			glVertex3fv((float *)&vs[acc]);
			acc++;
		}
		glEnd();
	}
}

void BaseDrawer::stripes(AdaptableStripeBuffer * data, const Vector3F & eyeDirection) const
{
	const unsigned ns = data->numStripe();
	unsigned * ncv = data->numCvs();
	Vector3F * pos = data->pos();
	Vector3F * col = data->col();
	float * w = data->width();
	Vector3F seg, nseg, straggle, q, q0, q1;
	glBegin(GL_QUADS);
	for(unsigned i = 0; i < ns; i++) {
		for(unsigned j = 1; j < ncv[i] - 2; j++) {
			seg = pos[j+1] - pos[j];
			nseg = seg.normal();
			straggle = nseg.cross(eyeDirection);
			straggle.normalize();
			
			glColor3fv((float *)&col[j]);
			
			if(j< 2) {
				q = pos[j] - straggle * w[j];
				glVertex3fv((float *)&q);
			
				q = pos[j] + straggle * w[j];
				glVertex3fv((float *)&q);
			}
			else {
				glVertex3fv((float *)&q0);
				glVertex3fv((float *)&q1);
			}
			
			glColor3fv((float *)&col[j+1]);
			
			q = pos[j+1] + straggle * w[j+1];
			glVertex3fv((float *)&q);
			
			q1 = q;
			
			q = pos[j+1] - straggle * w[j+1];
			glVertex3fv((float *)&q);
			
			q0 = q;
		}
		pos += ncv[i];
		col += ncv[i];
		w += ncv[i];
	}
	glEnd();
}
//:~
