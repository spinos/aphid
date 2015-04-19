#include <QtCore>
#include "SolverThread.h"
#include <CudaCSRMatrix.h>
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <CudaBase.h>
#include <cuConjugateGradient_implement.h>
#include <CudaDbgLog.h>
// Poisson ratio: transverse contraction strain to longitudinal extension 
// strain in the direction of stretching force. Tensile deformation is 
// considered positive and compressive deformation is considered negative. 
float nu = 0.33f; 
// Young modulus: also known as the tensile modulus or elastic modulus, 
// is a measure of the stiffness of an elastic material
float Y = 500000.0f;

float d15 = Y / (1.0f + nu) / (1.0f - 2 * nu);
float d16 = (1.0f - nu) * d15;
float d17 = nu * d15;
float d18 = Y / 2 / (1.0f + nu);
Vector3F D(d16, d17, d18); //Isotropic elasticity matrix D

float creep = 0.20f;
float yield = 0.04f;
float mass_damping=0.4f;
float m_max = 0.2f;
Vector3F gravity(0.0f,-9.81f,0.0f);

CudaDbgLog dbglg("stiffness.txt");

bool bUseStiffnessWarping = true;
#define TESTBLOCK 0
#define SOLVEONGPU 1
SolverThread::SolverThread(QObject *parent)
    : BaseSolverThread(parent)
{
    m_mesh = new FEMTetrahedronMesh;
#if TESTBLOCK
    m_mesh->generateBlocks(64,2,2, .5f, .5f, .5f);
    m_mesh->setDensity(30.f);
#else
    m_mesh->generateFromFile();
    m_mesh->setDensity(4.f);
#endif
    
    unsigned totalPoints = m_mesh->numPoints();
	
	ConjugateGradientSolver::init(totalPoints);
    
    m_K_row = new MatrixMap[totalPoints];
	m_V = new Vector3F[totalPoints];
	m_F = new Vector3F[totalPoints]; 
	m_F0 = new Vector3F[totalPoints]; 
	
	int * fixed = isFixed();
	unsigned i;
	for(i=0; i < totalPoints; i++) m_V[i].setZero();
	
	Vector3F * Xi = m_mesh->Xi();
	for(i=0; i < totalPoints; i++) {
	    if(i==7 || i==5 || i==8 || i==18)
	    //if(Xi[i].x<.1f)
            fixed[i]=1;
        else
            fixed[i]=0;   
	}
	
	calculateK();
	clearStiffnessAssembly();
	m_mesh->recalcMassMatrix(fixed);
	initializePlastic();
	
	qDebug()<<"num points "<<m_mesh->numPoints();
	qDebug()<<"total mass "<<m_mesh->mass();
	qDebug()<<"num tetrahedrons "<<m_mesh->numTetrahedra();
	qDebug()<<"total volume "<<m_mesh->volume0();
	
	m_stiffnessMatrix = new CudaCSRMatrix;
	m_hostK = new BaseBuffer;
}

SolverThread::~SolverThread() 
{
    std::cout<<" solver exit\n";
    cudaThreadExit();
    
    delete m_hostK;
    delete m_stiffnessMatrix;
}

void SolverThread::stepPhysics(float dt)
{
	computeForces();

	clearStiffnessAssembly();	

	if(bUseStiffnessWarping)
		updateOrientation();
	else
		resetOrientation();

	stiffnessAssembly();
	
	// addPlasticityForce(dt);
 
	dynamicsAssembly(dt);
 
#if SOLVEONGPU
	solveGpu(m_V, m_stiffnessMatrix);
#else
    solve(m_V);
#endif
 
	updatePosition(dt);
	
	dbglg.write("Re");
	unsigned totalTetrahedra = m_mesh->numTetrahedra();
	FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
	for(unsigned k=0;k<totalTetrahedra;k++) {
	    dbglg.write(k);
		dbglg.write(tetrahedra[k].Re.str());
	}
	
	dbglg.write("F0");
	unsigned totalPoints = m_mesh->numPoints();
	for(unsigned k=0;k<totalPoints;k++) {
	    dbglg.write(k);
		dbglg.write(m_F0[k].str());
	}
 
	dbglg.writeMat33(m_stiffnessMatrix->valueBuf(), 
	    m_stiffnessMatrix->numNonZero(),
	    "K ");

	// groundCollision();
	// qDebug()<<"total volume "<<m_mesh->volume();
}

FEMTetrahedronMesh * SolverThread::mesh() { return m_mesh; }

void SolverThread::initOnDevice()
{
    CudaBase::SetDevice();
    stiffnessAssembly();
    unsigned totalPoints = m_mesh->numPoints();
    CSRMap vertexConnection;
    unsigned i=0;
    for(unsigned k=0;k<totalPoints;k++) {
		MatrixMap tmp = m_K_row[k];
		MatrixMap::iterator Kbegin = tmp.begin();
        MatrixMap::iterator Kend   = tmp.end();
		for (MatrixMap::iterator K = Kbegin; K != Kend;++K)
		{
            unsigned j  = K->first;
			vertexConnection[k*totalPoints + j]=i;
			i++;
		}
	}
	
	m_stiffnessMatrix->create(CSRMatrix::tMat33, totalPoints, vertexConnection);
	// m_stiffnessMatrix->verbose();
	std::cout<<"max nnz per row "<<m_stiffnessMatrix->maxNNZRow();
	m_stiffnessMatrix->initOnDevice();
	
	m_hostK->create(m_stiffnessMatrix->numNonZero() * 36);
	
	ConjugateGradientSolver::initOnDevice();
}

void SolverThread::calculateK()
{
    unsigned totalTetrahedra = m_mesh->numTetrahedra();
    Vector3F * Xi = m_mesh->Xi();
    FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
    
    for(unsigned k=0;k<totalTetrahedra;k++) {
		
		Vector3F x0 = Xi[tetrahedra[k].indices[0]];
		Vector3F x1 = Xi[tetrahedra[k].indices[1]];
		Vector3F x2 = Xi[tetrahedra[k].indices[2]];
		Vector3F x3 = Xi[tetrahedra[k].indices[3]];
		
		//For this check page no.: 344-346 of Kenny Erleben's book Physics based Animation
		//Eq. 10.30(a-c)
		Vector3F e10 = x1-x0;
		Vector3F e20 = x2-x0;
		Vector3F e30 = x3-x0;

		// tetrahedra[k].e1 = e10;
		// tetrahedra[k].e2 = e20;
		// tetrahedra[k].e3 = e30;

		tetrahedra[k].volume= FEMTetrahedronMesh::getTetraVolume(e10,e20,e30);
		
		//Eq. 10.32
		Matrix33F E; 
		E.fill(e10, e20, e30);
		
		float detE = E.determinant(); if(detE ==0.f) std::cout<<" zero det "<<E.str()<<"\n";
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

		//Eq. 10.43 
		//Bn ~ [bn cn dn]^T
		// bn = d/dx N0 = [ invE00 invE10 invE20 invE30 ]
		// cn = d/dy N0 = [ invE01 invE11 invE21 invE31 ]
		// dn = d/dz N0 = [ invE02 invE12 invE22 invE32 ]
		tetrahedra[k].B[0] = Vector3F(invE00, invE01, invE02);		
		tetrahedra[k].B[1] = Vector3F(invE10, invE11, invE12);		
		tetrahedra[k].B[2] = Vector3F(invE20, invE21, invE22);		
		tetrahedra[k].B[3] = Vector3F(invE30, invE31, invE32);
		
		// std::cout<<"B[0] "<<tetrahedra[k].B[0]<<"\n";
		// std::cout<<"B[1] "<<tetrahedra[k].B[1]<<"\n";
		// std::cout<<"B[2] "<<tetrahedra[k].B[2]<<"\n";
		// std::cout<<"B[3] "<<tetrahedra[k].B[3]<<"\n";
 
		for(unsigned i=0;i<4;i++) {
			for(unsigned j=0;j<4;j++) {
				Matrix33F & Ke = tetrahedra[k].Ke[i][j];
				float d19 = tetrahedra[k].B[i].x;
				float d20 = tetrahedra[k].B[i].y;
				float d21 = tetrahedra[k].B[i].z;
				float d22 = tetrahedra[k].B[j].x;
				float d23 = tetrahedra[k].B[j].y;
				float d24 = tetrahedra[k].B[j].z;
				*Ke.m(0, 0)= d16 * d19 * d22 + d18 * (d20 * d23 + d21 * d24);
				*Ke.m(0, 1)= d17 * d19 * d23 + d18 * (d20 * d22);
				*Ke.m(0, 2)= d17 * d19 * d24 + d18 * (d21 * d22);

				*Ke.m(1, 0)= d17 * d20 * d22 + d18 * (d19 * d23);
				*Ke.m(1, 1)= d16 * d20 * d23 + d18 * (d19 * d22 + d21 * d24);
				*Ke.m(1, 2)= d17 * d20 * d24 + d18 * (d21 * d23);

				*Ke.m(2, 0)= d17 * d21 * d22 + d18 * (d19 * d24);
				*Ke.m(2, 1)= d17 * d21 * d23 + d18 * (d20 * d24);
				*Ke.m(2, 2)= d16 * d21 * d24 + d18 * (d20 * d23 + d19 * d22);

				Ke *= tetrahedra[k].volume;
				
				// qDebug()<<Ke.str().c_str();
			}
		}
 	}
}

void SolverThread::clearStiffnessAssembly() 
{	 
    unsigned totalPoints = m_mesh->numPoints();
	for(unsigned k=0;k<totalPoints;k++) {
		m_F0[k].setZero();
		
		for (MatrixMap::iterator Kij = m_K_row[k].begin() ; Kij != m_K_row[k].end(); ++Kij )
			Kij->second.setZero();
	}
}

void SolverThread::initializePlastic() 
{
    FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
	unsigned totalTetrahedra = m_mesh->numTetrahedra();
	for(unsigned i=0;i<totalTetrahedra;i++) {
		for(int j=0;j<6;j++) 
			tetrahedra[i].plastic[j]=0;		
	} 
}

void SolverThread::computeForces() 
{
    int * fixed = isFixed();
	unsigned i=0; 
	unsigned totalPoints = m_mesh->numPoints();
	float * mass = m_mesh->M();
	for(i=0;i<totalPoints;i++) {
		m_F[i].setZero();
		if(fixed[i] < 1)
		//add gravity force only for non-fixed points
		    m_F[i] += gravity * mass[i];
	}
}

void SolverThread::updateOrientation()
{
    FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
	unsigned totalTetrahedra = m_mesh->numTetrahedra();
	Vector3F * X = m_mesh->X();
	Vector3F * Xi = m_mesh->Xi();
	for(unsigned k=0;k<totalTetrahedra;k++) {
		//Based on description on page 362-364 
		float div6V = 1.0f / tetrahedra[k].volume*6.0f;

		unsigned i0 = tetrahedra[k].indices[0];
		unsigned i1 = tetrahedra[k].indices[1];
		unsigned i2 = tetrahedra[k].indices[2];
		unsigned i3 = tetrahedra[k].indices[3];

		Vector3F p0 = X[i0];
		Vector3F p1 = X[i1];
		Vector3F p2 = X[i2];
		Vector3F p3 = X[i3];

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
		// e1 = tetrahedra[k].e1;
		// e2 = tetrahedra[k].e2;
		// e3 = tetrahedra[k].e3;
		p0 = Xi[i0];
		p1 = Xi[i1];
		p2 = Xi[i2];
		p3 = Xi[i3];

		e1 = p1-p0;
		e2 = p2-p0;
		e3 = p3-p0;

		//Based on Eq. 10.133		
		Matrix33F &Re = tetrahedra[k].Re;
		*Re.m(0, 0) = e1.x * n1.x + e2.x * n2.x + e3.x * n3.x;  
		*Re.m(1, 0) = e1.x * n1.y + e2.x * n2.y + e3.x * n3.y;   
		*Re.m(2, 0) = e1.x * n1.z + e2.x * n2.z + e3.x * n3.z;

        *Re.m(0, 1) = e1.y * n1.x + e2.y * n2.x + e3.y * n3.x;  
		*Re.m(1, 1) = e1.y * n1.y + e2.y * n2.y + e3.y * n3.y;   
		*Re.m(2, 1) = e1.y * n1.z + e2.y * n2.z + e3.y * n3.z;

        *Re.m(0, 2) = e1.z * n1.x + e2.z * n2.x + e3.z * n3.x;  
		*Re.m(1, 2) = e1.z * n1.y + e2.z * n2.y + e3.z * n3.y;  
		*Re.m(2, 2) = e1.z * n1.z + e2.z * n2.z + e3.z * n3.z;
		
		Re.orthoNormalize();
		
	}
}

void SolverThread::resetOrientation() {	
    unsigned totalTetrahedra = m_mesh->numTetrahedra();
	FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
	for(unsigned k=0;k<totalTetrahedra;k++) {
		tetrahedra[k].Re.setIdentity();
	}
}

void SolverThread::updateF0()
{
    Vector3F * Xi = m_mesh->Xi();
	FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
	unsigned totalTetrahedra = m_mesh->numTetrahedra();
	for(unsigned k=0;k<totalTetrahedra;k++) {
		Matrix33F Re = tetrahedra[k].Re;
		Matrix33F ReT = Re; ReT.transpose();

		for (unsigned i = 0; i < 4; ++i) {
			//Based on pseudocode given in Fig. 10.11 on page 361
			Vector3F f(0.0f,0.0f,0.0f);
			for (unsigned j = 0; j < 4; ++j) {
				Matrix33F tmpKe = tetrahedra[k].Ke[i][j];
				Vector3F x0 = Xi[tetrahedra[k].indices[j]];
				Vector3F prod = tmpKe * x0;
				f += prod;				   
			}
			unsigned idx = tetrahedra[k].indices[i];
			m_F0[idx] -= Re*f;		
		}  	
	} 
}

void SolverThread::stiffnessAssembly() 
{ 
    updateF0();
    
    FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
	unsigned totalTetrahedra = m_mesh->numTetrahedra();
	for(unsigned k=0;k<totalTetrahedra;k++) {
		Matrix33F Re = tetrahedra[k].Re;
		Matrix33F ReT = Re; ReT.transpose();

		for (unsigned i = 0; i < 4; ++i) {
			for (unsigned j = 0; j < 4; ++j) {
				Matrix33F tmpKe = tetrahedra[k].Ke[i][j];
				if (j >= i) {
					//Based on pseudocode given in Fig. 10.12 on page 361
					Matrix33F tmp = (Re*tmpKe)*ReT; 
					Matrix33F tmpT = tmp; tmpT.transpose();
					int index = tetrahedra[k].indices[i]; 		
					 
					m_K_row[index][tetrahedra[k].indices[j]]+=(tmp);
					
					if (j > i) {
						index = tetrahedra[k].indices[j];
						m_K_row[index][tetrahedra[k].indices[i]]+= tmpT;
					}
				}

			}		
		}  	
	} 
}

void SolverThread::addPlasticityForce(float dt) 
{
    unsigned totalTetrahedra = m_mesh->numTetrahedra();
	Vector3F * X = m_mesh->X();
    Vector3F * Xi = m_mesh->Xi();
    FEMTetrahedronMesh::Tetrahedron * tetrahedra = m_mesh->tetrahedra();
	
    for(unsigned k=0;k<totalTetrahedra;k++) {
		float e_total[6];
		float e_elastic[6];
		for(int i=0;i<6;++i)
			e_elastic[i] = e_total[i] = 0;

		//--- Compute total strain: e_total  = Be (Re^{-1} x - x0)
		for(unsigned int j=0;j<4;++j) {

			Vector3F x_j  =  X[tetrahedra[k].indices[j]];
			Vector3F x0_j = Xi[tetrahedra[k].indices[j]];
			Matrix33F ReT  = tetrahedra[k].Re; ReT.transpose();
			Vector3F prod = ReT * x_j;
			//Vector3F(ReT[0][0]*x_j.x+ ReT[0][1]*x_j.y+ReT[0][2]*x_j.z, //tmpKe*x0;
			//						   ReT[1][0]*x_j.x+ ReT[1][1]*x_j.y+ReT[1][2]*x_j.z,
				//					   ReT[2][0]*x_j.x+ ReT[2][1]*x_j.y+ReT[2][2]*x_j.z);
				
			Vector3F tmp = prod - x0_j;

			//B contains Jacobian of shape funcs. B=SN
			float bj = tetrahedra[k].B[j].x;
			float cj = tetrahedra[k].B[j].y;
			float dj = tetrahedra[k].B[j].z;

			e_total[0] += bj*tmp.x;
			e_total[1] +=            cj*tmp.y;
			e_total[2] +=                       dj*tmp.z;
			e_total[3] += cj*tmp.x + bj*tmp.y;
			e_total[4] += dj*tmp.x            + bj*tmp.z;
			e_total[5] +=            dj*tmp.y + cj*tmp.z;
		}

		//--- Compute elastic strain
		for(int i=0;i<6;++i)
			e_elastic[i] = e_total[i] - tetrahedra[k].plastic[i];

		//--- if elastic strain exceeds c_yield then it is added to plastic strain by c_creep
		float norm_elastic = 0;
		for(int i=0;i<6;++i)
			norm_elastic += e_elastic[i]*e_elastic[i];
		norm_elastic = sqrt(norm_elastic);
		if(norm_elastic > yield) {
		    float creepdt = 1.f /dt;
		    if(creepdt > creep) creepdt = creep;
			float amount = dt * creepdt;  //--- make sure creep do not exceed 1/dt
			for(int i=0;i<6;++i)
				tetrahedra[k].plastic[i] += amount*e_elastic[i];
		}

		//--- if plastic strain exceeds c_max then it is clamped to maximum magnitude
		float norm_plastic = 0;
		for(int i=0;i<6;++i)
			norm_plastic += tetrahedra[k].plastic[i]* tetrahedra[k].plastic[i];
		norm_plastic = sqrt(norm_plastic);

		if(norm_plastic > m_max) { 
			float scale = m_max/norm_plastic;
			for(int i=0;i<6;++i)
				tetrahedra[k].plastic[i] *= scale;
		}

		for(unsigned n=0;n<4;++n) {
			float* e_plastic = tetrahedra[k].plastic;
			//bn, cn and dn are the shape function derivative wrt. x,y and z axis
			//These were calculated in CalculateK function

			//Eq. 10.140(a) & (b) on page 365
			float bn = tetrahedra[k].B[n].x;
			float cn = tetrahedra[k].B[n].y;
			float dn = tetrahedra[k].B[n].z;
			float D0 = D.x;
			float D1 = D.y;
			float D2 = D.z;
			Vector3F f  = Vector3F::Zero;

			float  bnD0 = bn*D0;
			float  bnD1 = bn*D1;
			float  bnD2 = bn*D2;
			float  cnD0 = cn*D0;
			float  cnD1 = cn*D1;
			float  cnD2 = cn*D2;
			float  dnD0 = dn*D0;
			float  dnD1 = dn*D1;
			float  dnD2 = dn*D2;
			
			//Eq. 10.141 on page 365
			f.x = bnD0*e_plastic[0] + bnD1*e_plastic[1] + bnD1*e_plastic[2] + cnD2*e_plastic[3] + dnD2*e_plastic[4];
			f.y = cnD1*e_plastic[0] + cnD0*e_plastic[1] + cnD1*e_plastic[2] + bnD2*e_plastic[3] +                  + dnD2*e_plastic[5];
			f.z = dnD1*e_plastic[0] + dnD1*e_plastic[1] + dnD0*e_plastic[2] +                    bnD2*e_plastic[4] + cnD2*e_plastic[5];
			
			f *= tetrahedra[k].volume;
			int idx = tetrahedra[k].indices[n];
			m_F[idx] += tetrahedra[k].Re*f;
		}
	}
}

void SolverThread::updateB(float dt)
{
    unsigned totalPoints = m_mesh->numPoints();
	Vector3F * X = m_mesh->X();
    float * mass = m_mesh->M();
	Vector3F * b = rightHandSide();
	for(unsigned k=0;k<totalPoints;k++) {

		float m_i = mass[k];
		b[k].setZero();
		
		MatrixMap tmp = m_K_row[k];
		MatrixMap::iterator Kbegin = tmp.begin();
        MatrixMap::iterator Kend   = tmp.end();
		for (MatrixMap::iterator K = Kbegin; K != Kend;++K)
		{
            unsigned j  = K->first;
			Matrix33F K_ij  = K->second; 
			Vector3F x_j   = X[j];	
			Vector3F prod = K_ij * x_j; 
			b[k] -= prod;
		}
	  
		b[k] -= m_F0[k];
		b[k] += m_F[k];
		b[k] *= dt;
		b[k] += m_V[k]*m_i;
	} 
}

void SolverThread::dynamicsAssembly(float dt) 
{
    updateB(dt);
	float dt2 = dt*dt;
	unsigned totalPoints = m_mesh->numPoints();
	float * mass = m_mesh->M();
	Matrix33F * hostK = (Matrix33F *)m_hostK->data();
	int i = 0;
	for(unsigned k=0;k<totalPoints;k++) {

		float m_i = mass[k];
		
		MatrixMap tmp = m_K_row[k];
		MatrixMap::iterator Kbegin = tmp.begin();
        MatrixMap::iterator Kend   = tmp.end();
		for (MatrixMap::iterator K = Kbegin; K != Kend;++K)
		{
            unsigned j  = K->first;
			Matrix33F K_ij  = K->second; 
			Matrix33F * A_ij = A(k,j);
 
			*A_ij = K_ij * dt2; 
			 
            if (k == j)
            {
              float c_i = mass_damping*m_i;
              float tmp = m_i + dt*c_i;
              *(*A_ij).m(0, 0) += tmp; 
			  *(*A_ij).m(1, 1) += tmp;  
			  *(*A_ij).m(2, 2) += tmp;
			}
			
			hostK[i] = *A_ij;
			i++;
		}
	}
	
	m_stiffnessMatrix->valueBuf()->hostToDevice(m_hostK->data());
}

void SolverThread::updatePosition(float dt) 
{
    unsigned totalPoints = m_mesh->numPoints();
    Vector3F * X = m_mesh->X();
    int * fixed = isFixed();
	for(unsigned k=0;k<totalPoints;k++) {
		if(fixed[k])
			continue;
		X[k] += m_V[k] * dt;
	}
}

void SolverThread::groundCollision()  
{
    unsigned totalPoints = m_mesh->numPoints();
    Vector3F * X = m_mesh->X();
    
	for(unsigned i=0;i<totalPoints;i++) {	
		if(X[i].y<0) //collision with ground
			X[i].y=0;
	}
}
//:~