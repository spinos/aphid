/*
 *  DrawDistanceField.h
 *  
 *
 *  Created by jian zhang on 7/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "ADistanceField.h"
#include "GeoDrawer.h"
#include <GridTables.h>
#include "tetrahedron_math.h"

namespace aphid {

class DrawDistanceField {

	float m_nodeColScl, m_nodeDrawSize;
	
public:
	DrawDistanceField();
	virtual ~DrawDistanceField();
	
protected:
	void setColorScale(const float & x);
	void setNodeDrawSize(const float & x);
	
	const float & nodeDrawSize() const;
	
	void drawNodes(const ADistanceField * fld, GeoDrawer * dr);
	void drawEdges(const ADistanceField * fld, GeoDrawer * dr);
	
	template<typename T>
	void drawGridCell(T * grd, GeoDrawer * dr)
	{
		dr->setColor(.15f, .15f, .15f);
		dr->boundingBox(grd->boundingBox() );
		Vector3F cellCol;
		BoundingBox cellBox;
		grd->begin();
		while(!grd->end() ) {
			
			sdb::gdt::GetCellColor(cellCol, grd->key().w );
			grd->getCellBBox(cellBox, grd->key() );
			cellBox.expand(-.04f - .04f * grd->key().w );
			
			dr->setColor(cellCol.x, cellCol.y, cellCol.z);
			dr->boundingBox(cellBox);
			
			//drawNode(grd->value(), dr, grd->key().w );
			
			grd->next();
		}
	}
	
	template<typename Tg, typename Tc>
	void drawGridNode(Tg * grd, GeoDrawer * dr)
	{
		grd->begin();
		while(!grd->end() ) {
			
			drawNodeInCell<Tc>(grd->value(), grd->key().w, dr );
			
			grd->next();
		}
	}
	
	template<typename T>
	void drawNodeInCell(T * cell, const float & level,
						GeoDrawer * dr)
	{
		float r, g, b;
		float nz = nodeDrawSize() * (1.f - .05f * level);
		
		cell->begin();
		while(!cell->end() ) {
			sdb::gdt::GetNodeColor(r, g, b,
						cell->value()->prop);
			dr->setColor(r, g, b);
			dr->cube(cell->value()->pos, nz );
			
			cell->next();
		}
	}
	
	template<typename T>
	void drawErrors(const ADistanceField * fld,
					sdb::Array<sdb::Coord2, T > * egs,
					const float & eps)
	{
		const DistanceNode * v = fld->nodes();
	
		glBegin(GL_LINES);
		egs->begin();
		while(!egs->end() ) {
			
			if(egs->value()->val > eps) {
				glVertex3fv((const float *)&v[egs->key().x].pos);
				glVertex3fv((const float *)&v[egs->key().y].pos);
			}
			
			egs->next();
		}
		glEnd();
	}
	
	template<typename Tn, typename Tt>
	bool checkTetraVolumeExt(const Tn * src,
							const int & ntet,
							const std::vector<Tt *> & tets) const
	{
		float mnvol = 1e20f, mxvol = -1e20f, vol;
		aphid::Vector3F p[4];
		int i = 0;
		for(;i<ntet;++i) {
			const Tt * t = tets[i];
			if(!t) continue;
			
			p[0] = src[t->iv0].pos;
			p[1] = src[t->iv1].pos;
			p[2] = src[t->iv2].pos;
			p[3] = src[t->iv3].pos;
			
			vol = tetrahedronVolume(p);
			if(mnvol > vol)
				mnvol = vol;
			if(mxvol < vol)
				mxvol = vol;
				
		}

		std::cout<<"\n min/max tetrahedron volume: "<<mnvol<<" / "<<mxvol;
		
		if(mnvol <= 0.f)
			std::cout<<"\n [ERROR] zero / negative volume";
			
		return mnvol > 0.f;
	}
	
};

}