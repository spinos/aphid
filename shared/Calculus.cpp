#include "Calculus.h"

namespace aphid {

namespace calc {

void legendreRules(int m, int n, float * v,
							float a, float b)
{
	const float dx = (b - a) / (m - 1);
  int i;
  int j;

  for ( i = 0; i < m; i++ )
  {
    v[i+0*m] = 1.0;
  }

  for ( i = 0; i < m; i++ )
  {
    v[i+1*m] = a + dx * i;
  }
 
  for ( j = 2; j <= n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      v[i+j*m] = ( ( float ) ( 2 * j - 1 ) *  (a + dx * i) * v[i+(j-1)*m]   
                 - ( float ) (     j - 1 ) *        v[i+(j-2)*m] ) 
                 / ( float ) (     j     );
    }
  }
}

float interpolate(int nknots, const float * yknots, int m, float x, float a, float b, float dx)
{
	int g = (x - a) / dx;
	if(g<1)
		return yknots[0];
	if(g>=m-1)
		return yknots[nknots-1];
		
	float movern =  ((float)m - 1.0) / ((float)nknots - 1);
	int n0 = g / movern;
	int n1 = n0 + 1;
	
	float alpha = (g - n0 * movern) / movern;
	return yknots[n0] * (1.f - alpha) + yknots[n1] * alpha;
}

float trapezIntegral(const float & a, const float & b, int m, const float * y)
{
	const float h = (b - a) / (m - 1);
	float c = h * .5f * (y[0] + y[m-1]);
	int i=1;
	for(;i<m-1;++i) {
		c += h * y[i];
	}
	return c;
}

/// 2 + 3 + 4 + 5 + 6
static const float GaussQuadratureWeight[20] = {
1.f, 1.f,
.555555556f, .888888889f, .555555556f,
.347854845f, .652145155f, .652145155f, .347854845f,
.236926885f, .478628670f, .568888889f, .478628670f, .236926885f,
.171324492f, .360761573f, .467913935f, .467913935f, .360761573f, .171324492f
};

static const float GaussQuadratureArgument[20] = {
-.577350269f, .577350269f,
-.774596669f, 0.f, .774596669f,
-.861136312f, -.339981044f, .339981044f, .861136312f,
-.906179846f, -.538469310f, 0.f, .538469310f, .906179846f,
-.932469514f, -.661209386f, -.238619186f, .238619186f, .661209386f, .932469514f
};

static const int GaussQuadratureNInd[7] = {
0, 0, 0, 
2,
5,
9,
14
};

void gaussQuadratureRule(int n, float * ci, float * xi)
{
	const int i0 = GaussQuadratureNInd[n];
	for(int i=0; i<n; ++i) {
		ci[i] = GaussQuadratureWeight[i0 + i];
		xi[i] = GaussQuadratureArgument[i0 + i];
	}
}

void tuple_next( int m1, int m2, int n, int *rank, int x[] )
{
  int i;
  int j;

  if ( m2 < m1 )
  {
    *rank = 0;
    return;
  }

  if ( *rank <= 0 )
  {
    for ( i = 0; i < n; i++ )
    {
      x[i] = m1;
    }
    *rank = 1;
  }
  else
  {
    *rank = *rank + 1;
    i = n - 1;

    for ( ; ; )
    {

      if ( x[i] < m2 )
      {
        x[i] = x[i] + 1;
        break;
      }

      x[i] = m1;

      if ( i == 0 )
      {
        *rank = 0;
        for ( j = 0; j < n; j++ )
        {
          x[j] = m1;
        }
        break;
      }
      i = i - 1;
    }
  }

  return;
}

}

}
