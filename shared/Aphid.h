/*
 *  Aphid.h
 *  lapl
 *
 *  Created by jian zhang on 3/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

inline char IsValueNearZero(const float & value)
{
	return (value < 10e-6 && value > -10e-6);
}

inline void SwapValues(float &a, float &b)
{
	float t = a;
	a = b;
	b = t;
}

inline const char *byte_to_binary(int x)
{
    static char b[33];
    b[32] = '\0';

    for (int z = 0; z < 32; z++) {
        b[31-z] = ((x>>z) & 0x1) ? '1' : '0';
    }

    return b;
}