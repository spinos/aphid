#include <QtCore>
#include "SolverThread.h"

float timeStep = 1.f / 60.f;

float nu = 0.33f;			//Poisson ratio
float Y = 500000.0f;		//Young modulus

float d15 = Y / (1.0f + nu) / (1.0f - 2 * nu);
float d16 = (1.0f - nu) * d15;
float d17 = nu * d15;
float d18 = Y / 2 / (1.0f + nu);

float density =1000.0f;
float creep = 0.20f;
float yield = 0.04f;
float mass_damping=1.0f;
float m_max = 0.2f;
Vector3F gravity(0.0f,-9.81f,0.0f); 

bool bUseStiffnessWarping = true;

SolverThread::SolverThread(QObject *parent)
    : QThread(parent)
{
    restart = false;
    abort = false;
    
    generateBlocks(10,3,3, 0.5f,0.5f,0.5f);
    
    m_mass = new float[m_totalPoints];
	
    m_A_row = new MatrixMap[m_totalPoints];
	m_K_row = new MatrixMap[m_totalPoints];
	m_b = new Vector3F[m_totalPoints];
	m_V = new Vector3F[m_totalPoints];
	m_F = new Vector3F[m_totalPoints]; 
	m_F0 = new Vector3F[m_totalPoints]; 
	m_residual = new Vector3F[m_totalPoints];
	m_update = new Vector3F[m_totalPoints];
	m_prev = new Vector3F[m_totalPoints];
	
	unsigned i;
	for(i=0; i < m_totalPoints; i++) m_V[i].setZero();
	
	calculateK();
	clearStiffnessAssembly();
	recalcMassMatrix();
	initializePlastic();
}

SolverThread::~SolverThread()
{
    mutex.lock();
    abort = true;
    condition.wakeOne();
    mutex.unlock();
    wait();
    
}

void SolverThread::simulate()
{
    QMutexLocker locker(&mutex);

    if (!isRunning()) {
        start(LowPriority);
    } else {
        restart = true;
        //qDebug()<<"wait";
        condition.wakeOne();
    }
}

void SolverThread::run()
{
   forever {
        // qDebug()<<"run";
	
	//for(int i=0; i< 2; i++) {
	    if(restart) {
	        // qDebug()<<"restart ";
            // break;
	    }
	        
	    if (abort) {
            // destroySolverData();
            qDebug()<<"abort";
            return;
        }
        
	    stepPhysics(timeStep);
	//}
 
		//if (!restart) {
		    // qDebug()<<"end";
            
		    emit doneStep();
		//}

		mutex.lock();
		
        if (!restart)
            condition.wait(&mutex);
			
        restart = false;
        mutex.unlock();
   }
}

void SolverThread::stepPhysics(float dt)
{
    
	computeForces();

	clearStiffnessAssembly();	

	if(bUseStiffnessWarping)
		updateOrientation();
	else
		resetOrientation();
/*
	StiffnessAssembly();
 
	AddPlasticityForce(dt);
 
	DynamicsAssembly(dt);
 
	ConjugateGradientSolver(dt);
 
	UpdatePosition(dt);

	GroundCollision();
*/
}

void SolverThread::generateBlocks(unsigned xdim, unsigned ydim, unsigned zdim, float width, float height, float depth)
{
    m_totalPoints = (xdim+1)*(ydim+1)*(zdim+1);
	m_X = new Vector3F[m_totalPoints];
	m_Xi= new Vector3F[m_totalPoints];
	m_IsFixed= new bool[m_totalPoints];
	int ind=0;
	float hzdim = zdim/2.0f;
	for(unsigned x = 0; x <= xdim; ++x) {
		for (unsigned y = 0; y <= ydim; ++y) {
			for (unsigned z = 0; z <= zdim; ++z) {			  
				m_X[ind] = Vector3F(width*x, height*z, depth*y);            
				m_Xi[ind] = m_X[ind];
				 
				//Make the first few points fixed
				if(m_Xi[ind].x<0.01)
					m_IsFixed[ind]=true;
				else
				    m_IsFixed[ind]=false;

				ind++;
			}
		}
	}
	//offset the m_tetrahedronl mesh by 0.5 units on y axis
	//and 0.5 of the depth in z axis
	for(unsigned i=0;i< m_totalPoints;i++) {
		m_X[i].y += 0.5;		
		m_X[i].z -= hzdim*depth; 
	}
	
	m_totalTetrahedrons = 5 * xdim * ydim * zdim;
	
	m_tetrahedron = new Tetrahedron[m_totalTetrahedrons];
	Tetrahedron * t = &m_tetrahedron[0];
	for (unsigned i = 0; i < xdim; ++i) {
		for (unsigned j = 0; j < ydim; ++j) {
			for (unsigned k = 0; k < zdim; ++k) {
				unsigned p0 = (i * (ydim + 1) + j) * (zdim + 1) + k;
				unsigned p1 = p0 + 1;
				unsigned p3 = ((i + 1) * (ydim + 1) + j) * (zdim + 1) + k;
				unsigned p2 = p3 + 1;
				unsigned p7 = ((i + 1) * (ydim + 1) + (j + 1)) * (zdim + 1) + k;
				unsigned p6 = p7 + 1;
				unsigned p4 = (i * (ydim + 1) + (j + 1)) * (zdim + 1) + k;
				unsigned p5 = p4 + 1;
				// Ensure that neighboring tetras are sharing faces
				if ((i + j + k) % 2 == 1) {
					addTetrahedron(t++, p1,p2,p6,p3);
					addTetrahedron(t++, p3,p6,p4,p7);
					addTetrahedron(t++, p1,p4,p6,p5);
					addTetrahedron(t++, p1,p3,p4,p0);
					addTetrahedron(t++, p1,p6,p4,p3); 
				} else {
					addTetrahedron(t++, p2,p0,p5,p1);
					addTetrahedron(t++, p2,p7,p0,p3);
					addTetrahedron(t++, p2,p5,p7,p6);
					addTetrahedron(t++, p0,p7,p5,p4);
					addTetrahedron(t++, p2,p0,p7,p5); 
				}
			}
		}
	}
	qDebug()<<"num points "<<m_totalPoints;
	qDebug()<<"num tetrahedrons "<<m_totalTetrahedrons;
	
}

void SolverThread::addTetrahedron(Tetrahedron *t, unsigned i0, unsigned i1, unsigned i2, unsigned i3) 
{
	t->indices[0]=i0;
	t->indices[1]=i1;
	t->indices[2]=i2;
	t->indices[3]=i3; 
}

unsigned SolverThread::numTetrahedrons() const { return m_totalTetrahedrons;}
Tetrahedron * SolverThread::tetrahedron() { return m_tetrahedron;}
Vector3F * SolverThread::X() { return m_X; }

float SolverThread::getTetraVolume(Vector3F e1, Vector3F e2, Vector3F e3) {
	return  e1.dot( e2.cross( e3 ) )/ 6.0f;
}

void SolverThread::calculateK()
{
    for(unsigned k=0;k<m_totalTetrahedrons;k++) {
		
		Vector3F x0 = m_Xi[m_tetrahedron[k].indices[0]];
		Vector3F x1 = m_Xi[m_tetrahedron[k].indices[1]];
		Vector3F x2 = m_Xi[m_tetrahedron[k].indices[2]];
		Vector3F x3 = m_Xi[m_tetrahedron[k].indices[3]];
		
		//For this check page no.: 344-346 of Kenny Erleben's book Physics based Animation
		//Eq. 10.30(a-c)
		Vector3F e10 = x1-x0;
		Vector3F e20 = x2-x0;
		Vector3F e30 = x3-x0;

		m_tetrahedron[k].e1 = e10;
		m_tetrahedron[k].e2 = e20;
		m_tetrahedron[k].e3 = e30;

		m_tetrahedron[k].volume= getTetraVolume(e10,e20,e30);
		
		//Eq. 10.32
		Matrix33F E; 
		E.fill(e10, e20, e30);
		
		float detE = E.determinant();
		float invDetE = 1.0f/detE;	
		
		//Eq. 10.40 (a) & Eq. 10.42 (a)
		//Shape function derivatives wrt x,y,z
		// d/dx N0
		float invE10 = (e20.z*e30.y - e20.y*e30.z)*invDetE;
		float invE20 = (e10.y*e30.z - e10.z*e30.y)*invDetE;
		float invE30 = (e10.z*e20.y - e10.y*e20.z)*invDetE;
		float invE00 = -invE10-invE20-invE30;

		//Eq. 10.40 (b) & Eq. 10.42 (b)
		// d/dy N0
		float invE11 = (e20.x*e30.z - e20.z*e30.x)*invDetE;
		float invE21 = (e10.z*e30.x - e10.x*e30.z)*invDetE;
		float invE31 = (e10.x*e20.z - e10.z*e20.x)*invDetE;
		float invE01 = -invE11-invE21-invE31;

		//Eq. 10.40 (c) & Eq. 10.42 (c)
		// d/dz N0
		float invE12 = (e20.y*e30.x - e20.x*e30.y)*invDetE;
		float invE22 = (e10.x*e30.y - e10.y*e30.x)*invDetE;
		float invE32 = (e10.y*e20.x - e10.x*e20.y)*invDetE;
		float invE02 = -invE12-invE22-invE32;

		Vector3F B[4];
		//Eq. 10.43 
		//Bn ~ [bn cn dn]^T
		// bn = d/dx N0 = [ invE00 invE10 invE20 invE30 ]
		// cn = d/dy N0 = [ invE01 invE11 invE21 invE31 ]
		// dn = d/dz N0 = [ invE02 invE12 invE22 invE32 ]
		m_tetrahedron[k].B[0] = Vector3F(invE00, invE01, invE02);		
		m_tetrahedron[k].B[1] = Vector3F(invE10, invE11, invE12);		
		m_tetrahedron[k].B[2] = Vector3F(invE20, invE21, invE22);		
		m_tetrahedron[k].B[3] = Vector3F(invE30, invE31, invE32);	 
 
		for(unsigned i=0;i<4;i++) {
			for(unsigned j=0;j<4;j++) {
				Matrix33F & Ke = m_tetrahedron[k].Ke[i][j];
				float d19 = m_tetrahedron[k].B[i].x;
				float d20 = m_tetrahedron[k].B[i].y;
				float d21 = m_tetrahedron[k].B[i].z;
				float d22 = m_tetrahedron[k].B[j].x;
				float d23 = m_tetrahedron[k].B[j].y;
				float d24 = m_tetrahedron[k].B[j].z;
				*Ke.m(0, 0)= d16 * d19 * d22 + d18 * (d20 * d23 + d21 * d24);
				*Ke.m(0, 1)= d17 * d19 * d23 + d18 * (d20 * d22);
				*Ke.m(0, 2)= d17 * d19 * d24 + d18 * (d21 * d22);

				*Ke.m(1, 0)= d17 * d20 * d22 + d18 * (d19 * d23);
				*Ke.m(1, 1)= d16 * d20 * d23 + d18 * (d19 * d22 + d21 * d24);
				*Ke.m(1, 2)= d17 * d20 * d24 + d18 * (d21 * d23);

				*Ke.m(2, 0)= d17 * d21 * d22 + d18 * (d19 * d24);
				*Ke.m(2, 1)= d17 * d21 * d23 + d18 * (d20 * d24);
				*Ke.m(2, 2)= d16 * d21 * d24 + d18 * (d20 * d23 + d19 * d22);

				Ke *= m_tetrahedron[k].volume;
				
				// qDebug()<<Ke.str().c_str();
			}
		}
 	}
}

void SolverThread::clearStiffnessAssembly() 
{	 
	for(unsigned k=0;k<m_totalPoints;k++) {
		m_F0[k].x=0.0f;
		m_F0[k].y=0.0f;
		m_F0[k].z=0.0f;
		
		for (MatrixMap::iterator Kij = m_K_row[k].begin() ; Kij != m_K_row[k].end(); ++Kij )
			Kij->second.setZero();
	}
}

void SolverThread::recalcMassMatrix() 
{
	//This is a lumped mass matrix
	//Based on Eq. 10.106 and pseudocode in Fig. 10.9 on page 358
	for(unsigned i=0;i<m_totalPoints;i++) {
		if(m_IsFixed[i])
			m_mass[i] = std::numeric_limits<float>::max();
		else
			m_mass[i] = 1.0f/m_totalPoints;
	}

	for(int i=0;i<m_totalTetrahedrons;i++) {
		float m = (density*m_tetrahedron[i].volume)* 0.25f;				 
		m_mass[m_tetrahedron[i].indices[0]] += m;
		m_mass[m_tetrahedron[i].indices[1]] += m;
		m_mass[m_tetrahedron[i].indices[2]] += m;
		m_mass[m_tetrahedron[i].indices[3]] += m;
	}	 
}

void SolverThread::initializePlastic() 
{
	for(unsigned i=0;i<m_totalTetrahedrons;i++) {
		for(int j=0;j<6;j++) 
			m_tetrahedron[i].plastic[j]=0;		
	} 
}

void SolverThread::computeForces() 
{
	unsigned i=0; 
	for(i=0;i<m_totalPoints;i++) {
		m_F[i].setZero();

		//add gravity force only for non-fixed points
		m_F[i] += gravity*m_mass[i];
	}
}

void SolverThread::updateOrientation()
{
	for(unsigned k=0;k<m_totalTetrahedrons;k++) {
		//Based on description on page 362-364 
		float div6V = 1.0f / m_tetrahedron[k].volume*6.0f;

		unsigned i0 = m_tetrahedron[k].indices[0];
		unsigned i1 = m_tetrahedron[k].indices[1];
		unsigned i2 = m_tetrahedron[k].indices[2];
		unsigned i3 = m_tetrahedron[k].indices[3];

		Vector3F p0 = m_X[i0];
		Vector3F p1 = m_X[i1];
		Vector3F p2 = m_X[i2];
		Vector3F p3 = m_X[i3];

		Vector3F e1 = p1-p0;
		Vector3F e2 = p2-p0;
		Vector3F e3 = p3-p0;

		//Eq. 10.129 on page 363 & Eq. 10.131 page 364
		//n1,n2,n3 approximate Einv
		Vector3F n1 = e2.cross(e3) * div6V;
		Vector3F n2 = e3.cross(e1) * div6V;
		Vector3F n3 = e1.cross(e2) * div6V;
		
		//Now get the rotation matrix from the initial undeformed (model/material coordinates)
		//We get the precomputed edge values
		e1 = m_tetrahedron[k].e1;
		e2 = m_tetrahedron[k].e2;
		e3 = m_tetrahedron[k].e3;

		//Based on Eq. 10.133		
		Matrix33F &Re = m_tetrahedron[k].Re;
		*Re.m(0, 0) = e1.x * n1.x + e2.x * n2.x + e3.x * n3.x;  
		*Re.m(0, 1) = e1.x * n1.y + e2.x * n2.y + e3.x * n3.y;   
		*Re.m(0, 2) = e1.x * n1.z + e2.x * n2.z + e3.x * n3.z;

        *Re.m(1, 0) = e1.y * n1.x + e2.y * n2.x + e3.y * n3.x;  
		*Re.m(1, 1) = e1.y * n1.y + e2.y * n2.y + e3.y * n3.y;   
		*Re.m(1, 2) = e1.y * n1.z + e2.y * n2.z + e3.y * n3.z;

        *Re.m(2, 0) = e1.z * n1.x + e2.z * n2.x + e3.z * n3.x;  
		*Re.m(2, 1) = e1.z * n1.y + e2.z * n2.y + e3.z * n3.y;  
		*Re.m(2, 2) = e1.z * n1.z + e2.z * n2.z + e3.z * n3.z;
		
		// ? m_tetrahedron[k].Re = ortho_normalize(m_tetrahedron[k].Re);
		
	}
}

void SolverThread::resetOrientation() {	
	for(unsigned k=0;k<m_totalTetrahedrons;k++) {
		m_tetrahedron[k].Re.setIdentity();
	}
}
