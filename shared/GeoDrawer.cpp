/*
 *  GeoDrawer.cpp
 *  aphid
 *
 *  Created by jian zhang on 1/10/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "GeoDrawer.h"
#include <BaseTransform.h>
#include <TransformManipulator.h>
#include <SkeletonJoint.h>
#include <SelectionArray.h>
#include <Anchor.h>
#include <GeodesicSphereMesh.h>
#include <PyramidMesh.h>
#include <CubeMesh.h>
#include <CircleCurve.h>
#include <DiscMesh.h>
#include <CurveBuilder.h>
#include <Geometry.h>
#include <GeometryArray.h>
#include <ATriangleMesh.h>
#include <ATetrahedronMesh.h>
#include <APointCloud.h>
#include <tetrahedron_math.h>

GeoDrawer::GeoDrawer() 
{
	m_sphere = new GeodesicSphereMesh(8);
	m_pyramid = new PyramidMesh;
	m_circle = new CircleCurve;
	m_cube = new CubeMesh;
	m_disc = new DiscMesh;
	m_alignDir = Vector3F::ZAxis;
}

GeoDrawer::~GeoDrawer()
{
	delete m_sphere;
	delete m_pyramid;
	delete m_circle;
	delete m_cube;
	delete m_disc;
}

void GeoDrawer::box(float width, float height, float depth)
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

void GeoDrawer::sphere(float size) const
{
	glPushMatrix();
	glScalef(size, size, size);
	drawMesh(m_sphere);
	glPopMatrix();
}

void GeoDrawer::cube(const Vector3F & p, const float & size) const
{
	glPushMatrix();

	glTranslatef(p.x-size*.5f, p.y-size*.5f, p.z-size*.5f);
	glScalef(size, size, size);
	
	drawMesh(m_cube);
	glPopMatrix();	
}

void GeoDrawer::solidCube(float x, float y, float z, float size)
{
	setWired(0);
	glPushMatrix();

	glTranslatef(x-size*.5f, y-size*.5f, z-size*.5f);
	glScalef(size, size, size);
	
	drawMesh(m_cube);
	glPopMatrix();	
}

void GeoDrawer::circleAt(const Vector3F & pos, const Vector3F & nor)
{
    glPushMatrix();
    Matrix44F mat;
    mat.setTranslation(pos);
    mat.setFrontOrientation(nor);
	useSpace(mat);
    linearCurve(*m_circle);
    glPopMatrix();
}

void GeoDrawer::circleAt(const Matrix44F & mat, float radius)
{
    glPushMatrix();
    useSpace(mat);
	m_circle->setRadius(radius);
    linearCurve(*m_circle);
    glPopMatrix();
}

void GeoDrawer::alignedCircle(const Vector3F & pos, float radius) const
{
	glPushMatrix();
    Matrix44F mat;
    mat.setTranslation(pos);
    mat.setFrontOrientation(m_alignDir);
	mat.scaleBy(radius);
	useSpace(mat);
	linearCurve(*m_circle);
    glPopMatrix();
}

void GeoDrawer::alignedDisc(const Vector3F & pos, float radius) const
{
	glPushMatrix();
    Matrix44F mat;
    mat.setTranslation(pos);
    mat.setFrontOrientation(m_alignDir);
	mat.scaleBy(radius);
	useSpace(mat);
	drawMesh(m_disc);
    glPopMatrix();
}

void GeoDrawer::arrow0(const Vector3F & at, const Vector3F & dir, float l, float w) const
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

void GeoDrawer::arrow(const Vector3F& origin, const Vector3F& dest) const
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

void GeoDrawer::arrow2(const Vector3F& origin, const Vector3F& dest, float width) const
{	
	Vector3F r = dest - origin;
	arrow0(dest, r, r.length() * .8f, width);
	
	r = origin - dest;
	arrow0(origin, r, r.length() * .2f, width);
}

void GeoDrawer::coordsys(float scale) const
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glColor3f(1.f, 0.f, 0.f);
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(scale, 0.f, 0.f));
	glColor3f(0.f, 1.f, 0.f);
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, scale, 0.f));
	glColor3f(0.f, 0.f, 1.f);
	arrow(Vector3F(0.f, 0.f, 0.f), Vector3F(0.f, 0.f, scale));
}

void GeoDrawer::coordsys(const Matrix33F & orient, float scale, Vector3F * p) const
{
	glPushMatrix();
	if(p) glTranslatef(p->x, p->y, p->z);
	float m[16];
	orient.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
	coordsys(scale);
	
	glPopMatrix();
}

void GeoDrawer::manipulator(TransformManipulator * m)
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

void GeoDrawer::spaceHandle(SpaceHandle * hand)
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

void GeoDrawer::anchor(Anchor *a, char active)
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
		CurveBuilder builder;
		
		for(unsigned i = 0; i < a->numPoints(); i++)
			builder.addVertex(a->getPoint(i)->p);
		builder.finishBuild(&curve);

		linearCurve(curve);
	}
	glPopMatrix();
}

void GeoDrawer::transform(BaseTransform * t) const
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

void GeoDrawer::skeletonJoint(SkeletonJoint * joint)
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

void GeoDrawer::moveHandle(int axis, bool active) const
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

void GeoDrawer::spinPlanes(BaseTransform * t) const
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

void GeoDrawer::spinHandle(TransformManipulator * m, bool active) const
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

void GeoDrawer::components(SelectionArray * arr)
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
		CurveBuilder builder;
		glDisable(GL_DEPTH_TEST);
		for(unsigned i = 0; i < numVert; i++) {
			Vector3F p = arr->getVertexP(i);
			solidCube(p.x, p.y, p.z, 0.2f);
			builder.addVertex(p);
		}
		glEnable(GL_DEPTH_TEST);
		
		if(arr->hasVertexPath()) {
		    builder.finishBuild(&curve);
			glLineWidth(2.f);
			linearCurve(curve);
			glLineWidth(1.f);
		}
    }
}

void GeoDrawer::primitive(Primitive * prim)
{
/*
	BaseMesh *geo = (BaseMesh *)prim->getGeometry();//printf("prim %i ", geo->entityType());
	const unsigned iface = prim->getComponentIndex();
	if(geo->isTriangleMesh())
		triangle((const BaseMesh *)geo, iface);
	else {
		patch((const BaseMesh *)geo, iface);
		}
*/
}

void GeoDrawer::drawDisc(float scale) const
{
	glPushMatrix();
	glScalef(scale, scale, scale);
	drawMesh(m_disc);
	glPopMatrix();
}

void GeoDrawer::drawSquare(const BoundingRectangle & b) const
{
	glBegin(GL_QUADS);
	glVertex3f(b.getMin(0), b.getMin(1), 0.f);
	glVertex3f(b.getMax(0), b.getMin(1), 0.f);
	glVertex3f(b.getMax(0), b.getMax(1), 0.f);
	glVertex3f(b.getMin(0), b.getMax(1), 0.f);
	glEnd();
}

void GeoDrawer::aabb(const Vector3F & low, const Vector3F & high) const
{
    glBegin(GL_LINES);

	glVertex3f(low.x,  low.y, low.z);
	glVertex3f(high.x, low.y, low.z);
	
	glVertex3f(low.x,  high.y, low.z);
	glVertex3f(high.x, high.y, low.z);
	
	glVertex3f(low.x,  low.y, high.z);
	glVertex3f(high.x, low.y, high.z);
	
	glVertex3f(low.x,  high.y, high.z);
	glVertex3f(high.x, high.y, high.z);
	
	glVertex3f(low.x,  low.y, low.z);
	glVertex3f(low.x, high.y, low.z);
	
	glVertex3f(high.x,  low.y, low.z);
	glVertex3f(high.x, high.y, low.z);
	
	glVertex3f(low.x,  low.y, high.z);
	glVertex3f(low.x, high.y, high.z);
	
	glVertex3f(high.x,  low.y, high.z);
	glVertex3f(high.x, high.y, high.z);
	
	glVertex3f(low.x,  low.y, low.z);
	glVertex3f(low.x,  low.y, high.z);
	
	glVertex3f(high.x,  low.y, low.z);
	glVertex3f(high.x,  low.y, high.z);
	
	glVertex3f(low.x,  high.y, low.z);
	glVertex3f(low.x,  high.y, high.z);
	
	glVertex3f(high.x, high.y, low.z);
	glVertex3f(high.x, high.y, high.z);
	glEnd();
}

void GeoDrawer::tetrahedron(const Vector3F * p) const
{
    glBegin(GL_LINES);
    vertex(p[0]);
	vertex(p[1]);
	
	vertex(p[1]);
	vertex(p[2]);
	
	vertex(p[2]);
	vertex(p[0]);
	
	vertex(p[0]);
	vertex(p[3]);
	
	vertex(p[1]);
	vertex(p[3]);
	
	vertex(p[2]);
	vertex(p[3]);
    glEnd();
}

void GeoDrawer::geometry(Geometry * geo) const
{
	if(geo->type() == TypedEntity::TGeometryArray) return geometryArray((GeometryArray *)geo);
	
	switch (geo->type()) {
		case TypedEntity::TBezierCurve:
			smoothCurve(*(BezierCurve *)geo, 4);
			break;
		case TypedEntity::TTriangleMesh:
			triangleMesh((ATriangleMesh *)geo);
			break;
		default:
			break;
	}
}

void GeoDrawer::geometryArray(GeometryArray * arr) const
{
	unsigned i = 0;
	for(;i<arr->numGeometries(); i++)
		geometry(arr->geometry(i));
}

void GeoDrawer::setAlignDir(const Vector3F & v)
{ m_alignDir = v; }

void GeoDrawer::pointCloud(APointCloud * cloud) const
{
	const unsigned n = cloud->numPoints();
	Vector3F * p = cloud->points();
	float * r = cloud->pointRadius();
	unsigned i = 0;
	for(; i<n;i++) alignedCircle(p[i], r[i]);
}

void GeoDrawer::tetrahedronMesh(ATetrahedronMesh * mesh) const
{
    const unsigned nt = mesh->numTetrahedrons();
    unsigned * indices = mesh->indices();
    Vector3F * points = mesh->points();
	glBegin(GL_TRIANGLES);
    unsigned i, j;
    Vector3F q;
    unsigned * tet;
    for(i=0; i< nt; i++) {
        tet = &indices[i*4];
        for(j=0; j< 12; j++) {
            q = points[ tet[ TetrahedronToTriangleVertex[j] ] ];
            glVertex3fv((GLfloat *)&q);
        }
    }
    glEnd();
}
//:~