#include <QtGui>
#include <QtOpenGL>
#include <BaseCamera.h>
#include "glwidget.h"
#include <KdTreeDrawer.h>
#include <BoxProgram.h>

#define NU 44
#define NV 33
#define NF (NU * NV)
#define NTri (NF * 2)
#define NI (NTri * 3)
#define NP ((NU + 1) * (NV + 1))

#define STRUCTURAL_SPRING 0
#define SHEAR_SPRING 1
#define BEND_SPRING 2

const float DEFAULT_DAMPING =  -0.0225f;
float	KsStruct = 10.75f,KdStruct = -0.25f;
float	KsShear = 50.75f,KdShear = -0.25f;
float	KsBend = 10.95f,KdBend = -0.24f;
float timeStep = 1.f/30.f;
Vector3F gravity(0.0f,-.0981f,0.0f);
float mass = 1.f;
float iniHeight = 30.f;

GLWidget::GLWidget(QWidget *parent) : Base3DView(parent)
{
    qDebug()<<"glview";
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	
	m_pos = new Vector3F[NP];
	m_posLast = new Vector3F[NP];
	m_force = new Vector3F[NP];
	m_indices = new unsigned[NI];
	
	unsigned i, j;
	unsigned c = 0;
	for(j=0; j<= NV; j++) {
		for(i=0; i <= NU; i++) {
			m_pos[c] = Vector3F((float)i, iniHeight, (float)j);
			m_posLast[c] = m_pos[c];
			c++;
		}
	}
/*
2 3
0 1
*/
	const unsigned nl = NU + 1;
	unsigned * id = &m_indices[0];
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			int i0 = j * nl + i;
			int i1 = i0 + 1;
			int i2 = i0 + nl;
			int i3 = i2 + 1;
			if ((j+i)%2) {
				*id++ = i0; *id++ = i2; *id++ = i1;
				*id++ = i1; *id++ = i2; *id++ = i3;
			} else {
				*id++ = i0; *id++ = i2; *id++ = i3;
				*id++ = i0; *id++ = i3; *id++ = i1;
			}
		}
	}
	
	m_numSpring = 0;
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU; i++) {
			m_numSpring++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i <= NU; i++) {
			m_numSpring++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			m_numSpring += 2;
		}
	}
	
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU - 1; i++) {
			m_numSpring++;
		}
	}
	
	for(j=0; j< NV - 1; j++) {
		for(i=0; i <= NU; i++) {
			m_numSpring++;
		}
	}
	
	qDebug()<<"num spring "<<m_numSpring;
	m_spring = new Spring[m_numSpring];
	
	Spring *spr = &m_spring[0];
	
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU; i++) {
			setSpring(spr, nl * j + i, nl * j + i + 1, KsStruct, KdStruct, STRUCTURAL_SPRING);
			spr++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i <= NU; i++) {
			setSpring(spr, nl * j + i, nl * (j + 1) + i, KsStruct, KdStruct, STRUCTURAL_SPRING);
			spr++;
		}
	}
	
	for(j=0; j< NV; j++) {
		for(i=0; i < NU; i++) {
			setSpring(spr, nl * j + i, nl * (j + 1) + i + 1, KsShear, KdShear, SHEAR_SPRING);
			spr++;
			setSpring(spr, nl * j + i + 1, nl * (j + 1) + i, KsShear, KdShear, SHEAR_SPRING);
			spr++;
		}
	}
	
	for(j=0; j<= NV; j++) {
		for(i=0; i < NU - 1; i++) {
			setSpring(spr, nl * j + i, nl * j + i + 2, KsBend, KdBend, BEND_SPRING);
			spr++;
		}
	}
	
	for(j=0; j< NV - 1; j++) {
		for(i=0; i <= NU; i++) {
			setSpring(spr, nl * j + i, nl * (j + 2) + i, KsBend, KdBend, BEND_SPRING);
			spr++;
		}
	}
	m_program = new BoxProgram;
	qDebug()<<"view..";
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
}

void GLWidget::clientInit()
{
    m_program->createCvs(NP);
    m_program->createIndices(NI, m_indices);
    m_program->createAabbs(NTri);
    QTimer *timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(simulate()));
	timer->start(33);
}

void GLWidget::clientDraw()
{
    // simulate();
    //qDebug()<<"dr";
    //return;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

	glColor3f(1,1,1);
	glBegin(GL_TRIANGLES);
	unsigned i;
	for(i=0; i< NI; i += 3) {
		Vector3F p1 = m_pos[m_indices[i]];
		Vector3F p2 = m_pos[m_indices[i+1]];
		Vector3F p3 = m_pos[m_indices[i+2]];
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
		glVertex3f(p3.x,p3.y,p3.z);
	}
	glEnd();
	
	
	Vector3F bbox[NTri * 2];
    m_program->getAabbs(bbox, NTri);

	GeoDrawer * dr = getDrawer();
	dr->setColor(0.f, 0.5f, 0.f);
	for(i=0; i< NTri; i++) {
	    BoundingBox bb;
	    bb.updateMin(bbox[i*2]);
	    bb.updateMax(bbox[i*2 + 1]);
	    dr->boundingBox(bb);
	}
	return;
	
	glColor3f(1,0,0);
	glBegin(GL_LINES);
	for(i=0; i< m_numSpring; i++) {
		// if(m_spring[i].type != BEND_SPRING) continue;
		Vector3F p1 = m_pos[m_spring[i].p1];
		Vector3F p2 = m_pos[m_spring[i].p2];
		glVertex3f(p1.x,p1.y,p1.z);
		glVertex3f(p2.x,p2.y,p2.z);
	}
	glEnd();
}
//! [7]

//! [9]
void GLWidget::clientSelect(Vector3F & origin, Vector3F & ray, Vector3F & hit)
{
}
//! [9]

void GLWidget::clientDeselect()
{
}

//! [10]
void GLWidget::clientMouseInput(Vector3F & stir)
{
}
//! [10]

void GLWidget::simulate()
{
    //setUpdatesEnabled(false);
	for(int i=0; i< 8; i++)stepPhysics(timeStep);
	//setUpdatesEnabled(true);
}

void GLWidget::keyPressEvent(QKeyEvent *e)
{
	Base3DView::keyPressEvent(e);
}

void GLWidget::keyReleaseEvent(QKeyEvent *event)
{
	Base3DView::keyReleaseEvent(event);
}

void GLWidget::setSpring(Spring * dest, unsigned a, unsigned b, float ks, float kd, int type) 
{
	dest->p1=a;
	dest->p2=b;
	dest->Ks=ks;
	dest->Kd=kd;
	dest->type = type;
	Vector3F deltaP = m_pos[a]-m_pos[b];
	dest->rest_length = deltaP.length();
}

void GLWidget::stepPhysics(float dt)
{
	computeForces(dt);
	integrateVerlet(dt);
	m_program->run(m_pos, NTri, NP);
}

void GLWidget::computeForces(float dt) 
{
	unsigned i=0;

	for(i=0;i< NP;i++) {
		m_force[i].setZero();
		Vector3F V = getVerletVelocity(m_pos[i], m_posLast[i], dt);
		//add gravity force
		if(i!=0 && i!=NU)	 
			m_force[i] += gravity*mass;
		//add force due to damping of velocity
		m_force[i] += V * DEFAULT_DAMPING;
	}


	for(i=0;i< m_numSpring; i++) {
		Spring & spring = m_spring[i];
		Vector3F p1 = m_pos[spring.p1];
		Vector3F p1Last = m_posLast[spring.p1];
		Vector3F p2 = m_pos[spring.p2];
		Vector3F p2Last = m_posLast[spring.p2];

		Vector3F v1 = getVerletVelocity(p1, p1Last, dt);
		Vector3F v2 = getVerletVelocity(p2, p2Last, dt);

		Vector3F deltaP = p1-p2;
		Vector3F deltaV = v1-v2;
		float dist = deltaP.length();

		float leftTerm = -spring.Ks * (dist - spring.rest_length);
		float rightTerm = spring.Kd * (deltaV.dot(deltaP)/dist);
		Vector3F springForce = deltaP.normal() * (leftTerm + rightTerm);

		if(spring.p1 != 0 && spring.p1 != NU)
			m_force[spring.p1] += springForce;
		if(spring.p2 != 0 && spring.p2 != NU )
			m_force[spring.p2] -= springForce;
	}
}

void GLWidget::integrateVerlet(float deltaTime) {
	float deltaTime2Mass = (deltaTime*deltaTime)/ mass;
	unsigned i=0;


	for(i=0;i< NP;i++) {
		Vector3F buffer = m_pos[i];

		m_pos[i] = m_pos[i] + (m_pos[i] - m_posLast[i]) + m_force[i] * deltaTime2Mass;
		//qDebug()<<m_force[i].x<<" "<<m_force[i].y<<" "<<m_force[i].z;
		m_posLast[i] = buffer;

		if(m_pos[i].y <0) {
			m_pos[i].y = 0;
		}
	}
}

Vector3F GLWidget::getVerletVelocity(Vector3F x_i, Vector3F xi_last, float dt ) 
{
	return  (x_i - xi_last) / dt;
}