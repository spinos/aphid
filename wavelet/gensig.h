/*
 *  gensig.h
 *  wvlt
 *
 *  Created by jian zhang on 9/17/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef WLA_GEN_SIG_H
#define WLA_GEN_SIG_H

#include <plots.h>

void gen1dsig(aphid::UniformPlot1D * line,
				int m);

/// M number of rows
/// N number of columns
/// P number of ranks
/// Noi weight of noise
void gen2dsig(aphid::UniformPlot2D * img,
				int m, int n, int p,
				float noi=0.5f);
/// M0 M1 range of rows
/// N0 N1 range of columns
void gen2dsigFrac(aphid::UniformPlot2D * img,
				int m, int n, int p,
				int m0, int n0, int m1, int n1);

#endif


