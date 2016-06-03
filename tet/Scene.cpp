/*
 *  Scene.cpp
 *  
 *
 *  Created by jian zhang on 6/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "Scene.h"
using namespace aphid;
namespace ttg {

Scene::Scene() {}
Scene::~Scene() {}

bool Scene::init() 
{ return true; }

const char * Scene::titleStr() const
{ return "unknown"; }

void Scene::draw(GeoDrawer * dr) {}

}