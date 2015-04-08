#ifndef BCC_COMMON_H
#define BCC_COMMON_H

static const float OctChildOffset[8][3] = {
{-1.f, -1.f, -1.f},
{-1.f, -1.f, 1.f},
{-1.f, 1.f, -1.f},
{-1.f, 1.f, 1.f},
{1.f, -1.f, -1.f},
{1.f, -1.f, 1.f},
{1.f, 1.f, -1.f},
{1.f, 1.f, 1.f}};

static const float HexHeighborOffset[6][3] = {
{0.f, 0.f, -1.f},
{0.f, 0.f, 1.f},
{0.f, -1.f, 0.f},
{0.f, 1.f, 0.f},
{-1.f,0.f, 0.f},
{1.f, 0.f, 0.f}};

static const float HexOctahedronOffset[6][6][3] = {
{{0.f,0.f,0.f},{ 0.f, 0.f,-1.f},{-.5f,-.5f,-.5f},{-.5f, .5f,-.5f},{ .5f,-.5f,-.5f},{ .5f, .5f,-.5f}},
{{0.f,0.f,0.f},{ 0.f, 0.f, 1.f},{-.5f, .5f, .5f},{-.5f,-.5f, .5f},{ .5f, .5f, .5f},{ .5f,-.5f, .5f}},
{{0.f,0.f,0.f},{ 0.f,-1.f, 0.f},{-.5f,-.5f, .5f},{-.5f,-.5f,-.5f},{ .5f,-.5f, .5f},{ .5f,-.5f,-.5f}},
{{0.f,0.f,0.f},{ 0.f, 1.f, 0.f},{-.5f, .5f,-.5f},{-.5f, .5f, .5f},{ .5f, .5f,-.5f},{ .5f, .5f, .5f}},
{{0.f,0.f,0.f},{-1.f, 0.f, 0.f},{-.5f,-.5f,-.5f},{-.5f,-.5f, .5f},{-.5f, .5f,-.5f},{-.5f, .5f, .5f}},
{{0.f,0.f,0.f},{ 1.f, 0.f, 0.f},{ .5f, .5f,-.5f},{ .5f, .5f, .5f},{ .5f,-.5f,-.5f},{ .5f,-.5f, .5f}}
};

/* 
     1
     |
   2------4
  /  |   /
 /   |  /
3------5
     |
     0
*/
static const int OctahedronToTetrahedronVetex[4][4] = {
{0, 5, 3, 1},
{0, 2, 4, 1},
{0, 3, 2, 1},
{0, 4, 5, 1}
};

#include <tetrahedron_math.h>

/*
*  6 + 8 connections
*  x-y plane
*                     3
*                 9   |   13
*                   \ | /
*                    \|/
*              4------1------5
*                    /|\
*                   / | \
*                 7   |   11
*                     2
*  z-y plane
*                     3
*                 8   |   9
*                   \ | /
*                    \|/
*              0------4------1
*                    /|\
*                   / | \
*                 6   |   7
*                     2
*/
#endif        //  #ifndef BCC_COMMON_H

