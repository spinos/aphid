/******************************************************************************
 * Created:	4/15/2011
 * Module:	constant
 * Purpose:	constant shaders for startup writers
 *
 * Exports:
 *      zhang_illum_constant
 *      zhang_illum_constant_version
 *
 *****************************************************************************/

#ifdef HPUX
#pragma OPT_LEVEL 1	/* workaround for HP/UX optimizer bug, +O2 and +O3 */
#endif

#include <stdio.h>
#include <stdlib.h>		/* for abs */
#include <float.h>		/* for FLT_MAX */
#include <math.h>
#include <string.h>
#include <assert.h>
#include "shader.h"
#include "mi_shader_if.h"

struct zhang_illum_constant_t {
	miColor		surfaceColor;
};


extern "C" DLLEXPORT int zhang_illum_constant_version(void) {return(1);}

extern "C" DLLEXPORT miBoolean zhang_illum_constant(
	miColor		*result,
	miState		*state,
	struct zhang_illum_constant_t *paras)
{ 
        /* check for illegal calls */
        if (state->type == miRAY_SHADOW || state->type == miRAY_DISPLACE ) {
		return(miFALSE);
	}

	*result    = *mi_eval_color(&paras->surfaceColor);	
	result->a  = 1;
	return(miTRUE);
}
