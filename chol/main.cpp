#include "cholmod.h"

int main (void) 
{ 
    cholmod_sparse *A ; 
    cholmod_dense *x, *b, *r ; 
    cholmod_factor *L ; 
    double one [2] = {1,0}, m1 [2] = {-1,0} ; /* basic scalars */ 
    cholmod_common c ; 
    cholmod_start (&c) ;
	A = cholmod_allocate_sparse(8, 8, 32, 0, 1, -1, CHOLMOD_REAL, &c);
	//A = cholmod_speye(8, 8, CHOLMOD_REAL, &c); 
	
	int	*colptr = (int*)(A->p);
	int	*rowind = (int*)(A->i);
	float *values = (float*)(A->x);
	
	colptr[0] = 0;
	for(int col =0; col < 8; col++) {
		rowind[colptr[col]] = col;
		values[colptr[col]] = 1.0;
		rowind[colptr[col]+1] = ( col == 0 ) ? 7 : col - 1;
		values[colptr[col]+1] = 1.0;
		colptr[col+1] = colptr[col] + 2;
	}

	cholmod_print_sparse (A, "A", &c) ;/*
	if (A == NULL || A->stype == 0) 
	{ 
		cholmod_free_sparse (&A, &c) ; 
		cholmod_finish (&c) ; 
		return 0; 
	} 
	b = cholmod_ones (A->nrow, 1, A->xtype, &c) ; 
	L = cholmod_analyze (A, &c) ;
	cholmod_factorize (A, L, &c) ; 
	x = cholmod_solve (CHOLMOD_A, L, b, &c) ; 
	r = cholmod_copy_dense (b, &c) ; 
	cholmod_sdmult (A, 0, m1, one, x, r, &c) ; 
	printf ("norm(b-Ax) %8.1e\n", 
	cholmod_norm_dense (r, 0, &c)) ;
	cholmod_free_factor (&L, &c) ; 
	cholmod_free_sparse (&A, &c) ; 
	cholmod_free_dense (&r, &c) ; 
	cholmod_free_dense (&x, &c) ; 
	cholmod_free_dense (&b, &c) ;
*/
	cholmod_finish (&c) ;
	return 0; 
} 

