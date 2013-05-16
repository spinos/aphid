/*
 *  patchTopology.cpp
 *  catmullclark
 *
 *  Created by jian zhang on 10/28/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <iostream>
#include <Vector3F.h>
#include "patchTopology.h"

PatchTopology::PatchTopology()
{

}

PatchTopology::~PatchTopology()
{

}

void PatchTopology::setVertexValence(unsigned* data)
{
	_valence = data;
}

void PatchTopology::setVertex(unsigned* data)
{
	_vertices = data;
}

void PatchTopology::setBoundary(char* data)
{
	_boundary = data;
}

int PatchTopology::getCornerIndex(int i) const
{
	int corner = 8;
	if(i == 1)
		corner = 9;
	else if(i == 2)
		corner = 15;
	else if(i == 3)
		corner = 14;
	return V(corner);
}

char PatchTopology::isCornerOnBoundary(int i) const
{
	if(i == 0)
	{
		if(!_boundary[2] || !_boundary[6])
			return 1;
		if(_valence[_vertices[8]] > 3)
		{
			if(!_boundary[1])
				return 1;
		}
	}
	else if(i == 1)
	{
		if(!_boundary[2] || !_boundary[8])
			return 1;
		if(_valence[_vertices[9]] > 3)
		{
			if(!_boundary[3])
				return 1;
		}
	}
	else if(i == 2)
	{
		if(!_boundary[12] || !_boundary[8])
			return 1;
		if(_valence[_vertices[15]] > 3)
		{
			if(!_boundary[13])
				return 1;
		}
	}
	else
	{
		if(!_boundary[12] || !_boundary[6])
			return 1;
		if(_valence[_vertices[14]] > 3)
		{
			if(!_boundary[11])
				return 1;
		}
	}
	return 0;
}

char PatchTopology::isEdgeOnBoundary(int i) const
{
	if(i == 0)
	{
		if(!_boundary[2])
			return 1;
	}
	else if(i == 1)
	{
		if(!_boundary[8])
			return 1;
	}
	else if(i == 2)
	{
		if(!_boundary[12])
			return 1;
	}
	else
	{
		if(!_boundary[6])
			return 1;
	}
	return 0;
}

void PatchTopology::getBoundaryEdgesOnCorner(int i, int* edge) const
{
	int end0 = -1, end1 = -1;
	if(i == 0)
	{
		if(!_boundary[2])
		{
			end0 = 9;
		}
		if(!_boundary[1])
		{
			if(end0 < 0) 
				end0 = 2;
		}
		if(!_boundary[6])
		{
			if(end0 < 0) 
				end0 = 7;
			end1 = 14;
		}
		if(!_boundary[1])
		{
			if(end1 < 0) 
				end1 = 7;
		}
		if(!_boundary[2])
		{
			if(end1 < 0) 
				end1 = 2;
		}
	}
	else if(i == 1)
	{
		if(!_boundary[8])
		{
			end0 = 15;
		}
		if(!_boundary[3])
		{
			if(end0 < 0) 
				end0 = 10;
		}
		if(!_boundary[2])
		{
			if(end0 < 0) 
				end0 = 3;
			end1 = 8;
		}
		if(!_boundary[3])
		{
			if(end1 < 0) 
				end1 = 3;
		}
		if(!_boundary[8])
		{
			if(end1 < 0) 
				end1 = 10;
		}
	}
	else if(i == 2)
	{
		if(!_boundary[12])
		{
			end0 = 14;
		}
		if(!_boundary[13])
		{
			if(end0 < 0) 
				end0 = 21;
		}
		if(!_boundary[8])
		{
			if(end0 < 0) 
				end0 = 16;
			end1 = 9;
		}
		if(!_boundary[13])
		{
			if(end1 < 0) 
				end1 = 16;
		}
		if(!_boundary[12])
		{
			if(end1 < 0) 
				end1 = 21;
		}
	}
	else
	{
		if(!_boundary[6])
		{
			end0 = 8;
		}
		if(!_boundary[11])
		{
			if(end0 < 0) 
				end0 = 13;
		}
		if(!_boundary[12])
		{
			if(end0 < 0) 
				end0 = 20;
			end1 = 15;
		}
		if(!_boundary[11])
		{
			if(end1 < 0) 
				end1 = 20;
		}
		if(!_boundary[6])
		{
			if(end1 < 0) 
				end1 = 13;
		}
	}
	edge[0] = V(end0);
	edge[1] = V(end1);
}

void PatchTopology::getFringeAndEdgesOnCorner(int i, int* fringe, int* edge) const
{
	int vl;
	if(i == 0)
	{
		vl = _valence[V(8)];
		if(vl == 3)
		{
			edge[0] = V(9);
			edge[1] = V(14);
			edge[2] = V(2);
			fringe[0] = V(15);	
			fringe[1] = V(3);
			fringe[2] = V(13);		
		}
		else
		{
			edge[0] = V(9);
			edge[1] = V(14);
			edge[2] = V(7);
			edge[3] = V(2);
			fringe[0] = V(15);	
			fringe[1] = V(13);
			fringe[2] = V(1);
			fringe[3] = V(3);
			if(vl > 4)
			{
				edge[4] = V(1);
				fringe[2] = V(0);
				fringe[4] = V(6);
			}
		}
	}
	else if(i == 1)
	{
		vl = _valence[V(9)];
		if(vl == 3)
		{
			edge[0] = V(8);
			edge[1] = V(15);
			edge[2] = V(3);
			fringe[0] = V(14);	
			fringe[1] = V(2);
			fringe[2] = V(16);
		}
		else
		{
			edge[0] = V(10);
			edge[1] = V(15);
			edge[2] = V(8);
			edge[3] = V(3);
			fringe[0] = V(4);	
			fringe[1] = V(16);
			fringe[2] = V(14);
			fringe[3] = V(2);
			if(vl > 4)
			{
				edge[4] = V(4);
				fringe[0] = V(5);
				fringe[4] = V(11);
			}
		}
	}
	else if(i == 2)
	{
		vl = _valence[V(15)];
		if(vl == 3)
		{
			edge[0] = V(9);
			edge[1] = V(14);
			edge[2] = V(21);
			fringe[0] = V(8);	
			fringe[1] = V(10);
			fringe[2] = V(20);
		}
		else
		{
			edge[0] = V(16);
			edge[1] = V(21);
			edge[2] = V(14);
			edge[3] = V(9);
			fringe[0] = V(10);	
			fringe[1] = V(22);
			fringe[2] = V(20);
			fringe[3] = V(8);
			if(vl > 4)
			{
				edge[4] = V(22);
				fringe[1] = V(23);
				fringe[4] = V(17);
			}
		}
	}
	else
	{
		vl = _valence[V(14)];
		if(vl == 3)
		{
			edge[0] = V(8);
			edge[1] = V(15);
			edge[2] = V(20);
			fringe[0] = V(9);	
			fringe[1] = V(7);
			fringe[2] = V(21);
		}
		else
		{
			edge[0] = V(15);
			edge[1] = V(20);
			edge[2] = V(13);
			edge[3] = V(8);
			fringe[0] = V(9);	
			fringe[1] = V(21);
			fringe[2] = V(19);
			fringe[3] = V(7);
			if(vl > 4)
			{
				edge[4] = V(19);
				fringe[2] = V(18);
				fringe[4] = V(12);
			}
		}
	}
}

void PatchTopology::getFringeAndEdgesOnEdgeBySide(int i, int side, int* fringe, int* edge) const
{
	getEdgeBySide(i, side, edge);
	if(i == 0)
	{
		if(side == 0)
		{
			fringe[0] = V(2);
			fringe[1] = V(14);
			fringe[2] = V(3);
			fringe[3] = V(15);
		}
		else
		{
			fringe[0] = V(3);
			fringe[1] = V(15);
			fringe[2] = V(2);
			fringe[3] = V(14);
		}
	}
	else if(i == 1)
	{
		if(side == 0)
		{
			fringe[0] = V(10);
			fringe[1] = V(8);
			fringe[2] = V(16);
			fringe[3] = V(14);
		}
		else
		{
			fringe[0] = V(16);
			fringe[1] = V(14);
			fringe[2] = V(10);
			fringe[3] = V(8);
		}
	}
	else if(i == 2)
	{
		if(side == 0)
		{
			fringe[0] = V(21);
			fringe[1] = V(9);
			fringe[2] = V(20);
			fringe[3] = V(8);
		}
		else
		{
			fringe[0] = V(20);
			fringe[1] = V(8);
			fringe[2] = V(21);
			fringe[3] = V(9);
		}
	}
	else
	{
		if(side == 0)
		{
			fringe[0] = V(13);
			fringe[1] = V(15);
			fringe[2] = V(7);
			fringe[3] = V(9);
		}
		else
		{
			fringe[0] = V(7);
			fringe[1] = V(9);
			fringe[2] = V(13);
			fringe[3] = V(15);
		}
	}
}

void PatchTopology::getFringeOnFaceByCorner(int i, int* fringe) const
{
	if(i == 0)
	{
		fringe[0] = V(8);
		fringe[1] = V(9);
		fringe[2] = V(14);
		fringe[3] = V(15);
	}
	else if(i == 1)
	{
		fringe[0] = V(9);
		fringe[1] = V(8);
		fringe[2] = V(15);
		fringe[3] = V(14);
	}
	else if(i == 2)
	{
		fringe[0] = V(15);
		fringe[1] = V(9);
		fringe[2] = V(14);
		fringe[3] = V(8);
	}
	else
	{
		fringe[0] = V(14);
		fringe[1] = V(8);
		fringe[2] = V(15);
		fringe[3] = V(9);
	}
}

void PatchTopology::getEdgeBySide(int i, int side, int* edge) const
{
	if(i == 0)
	{
		if(side == 0)
		{
			edge[0] = V(8);
			edge[1] = V(9);
		}
		else
		{
			edge[0] = V(9);
			edge[1] = V(8);
		}
	}
	else if(i == 1)
	{
		if(side == 0)
		{
			edge[0] = V(9);
			edge[1] = V(15);
		}
		else
		{
			edge[0] = V(15);
			edge[1] = V(9);
		}
	}
	else if(i == 2)
	{
		if(side == 0)
		{
			edge[0] = V(15);
			edge[1] = V(14);
		}
		else
		{
			edge[0] = V(14);
			edge[1] = V(15);
		}
	}
	else
	{
		if(side == 0)
		{
			edge[0] = V(14);
			edge[1] = V(8);
		}
		else
		{
			edge[0] = V(8);
			edge[1] = V(14);
		}
	}
}

int PatchTopology::getValenceOnCorner(int i) const
{
	return _valence[getCornerIndex(i)];
}

int PatchTopology::getCornerValenceByEdgeBySide(int i, int side) const
{
	if(i == 0)
	{
		if(side == 0) return _valence[V(8)];
		else return _valence[V(9)];
	}
	else if(i == 1)
	{
		if(side == 0) return _valence[V(9)];
		else return _valence[V(15)];
	}
	else if(i == 2)
	{
		if(side == 0) return _valence[V(15)];
		else return _valence[V(14)];
	}
	else
	{
		if(side == 0) return _valence[V(14)];
		else return _valence[V(8)];
	}
}

int PatchTopology::V(int i) const
{
	return _vertices[i];
}

