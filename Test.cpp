//============================================================================
// Name        : Test.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "glut.h"
#include <math.h>
#include <map>
#include <vector>

using namespace std;

const static float M_PI = 3.141593f;



std::vector<Sphere> spheres;

struct PairIndices
{
  PairIndices()
  {
    index0 = index1 = -1;
  }
  PairIndices(int index0, int index1)
  {
    if(index0 < index1)
    {
      this->index0 = index0;
      this->index1 = index1;
    }else
    {
      this->index0 = index0;
      this->index1 = index1;
    }
  }
  bool operator < (const PairIndices &other) const
  {
    if(index0 < other.index0) return 1;
    if((index0 == other.index0) && (index1 < other.index1)) return 1;
    return 0;
  }

  int index0, index1;
};

bool enableWarmstarting = 1;

struct Collision
{
  Collision()
  {
    accumulatedImpulse = 0;
    active = 1;
  }
  PairIndices indices;
  vec2 point, normal;
  float desiredVelocityProjection;
  bool active;

  float accumulatedImpulse;


  void PreStep()
  {
    float bounce = 0.0f;
    float damping = 10.0f;
    float relativeVelocityProjection = (spheres[indices.index1].velocity - spheres[indices.index0].velocity) * normal;
    desiredVelocityProjection = Max(0, -bounce * relativeVelocityProjection - damping);

    if(enableWarmstarting)
    {
      spheres[indices.index0].ApplyImpulse(-normal * accumulatedImpulse);
      spheres[indices.index1].ApplyImpulse( normal * accumulatedImpulse);
    }
  }

  void Solve()
  {
    float relativeVelocityProjection = (spheres[indices.index1].velocity - spheres[indices.index0].velocity) * normal;
    float lambda = (desiredVelocityProjection - relativeVelocityProjection) / (spheres[indices.index0].invMass + spheres[indices.index1].invMass);
    if(!enableWarmstarting)
    {
      if(lambda > 0)
      {
        spheres[indices.index0].ApplyImpulse(-lambda * normal);
        spheres[indices.index1].ApplyImpulse( lambda * normal);
      }
    }else
    {
      if(accumulatedImpulse + lambda < 0)
      {
        lambda = -accumulatedImpulse;
      }
      accumulatedImpulse += lambda;

      spheres[indices.index0].ApplyImpulse(-lambda * normal);
      spheres[indices.index1].ApplyImpulse( lambda * normal);
    }
  }
};


typedef std::map<PairIndices, Collision> CollisionMap;
CollisionMap collisions;

void FindCollisions()
{
  for(size_t i = 0; i < spheres.size(); i++)
  {
    for(size_t j = i + 1; j < spheres.size(); j++)
    {
      if(spheres[i].isStatic && spheres[j].isStatic) continue;
      if((spheres[i].pos - spheres[j].pos).Length() < spheres[i].radius + spheres[j].radius)
      {
        PairIndices pairIndices(i, j);
        vec2 normal = (spheres[pairIndices.index1].pos - spheres[pairIndices.index0].pos).Direction();
        vec2 point = 0.5 * (
          spheres[pairIndices.index0].pos + normal * spheres[pairIndices.index0].radius +
          spheres[pairIndices.index1].pos - normal * spheres[pairIndices.index1].radius);

        CollisionMap::iterator it = collisions.find(pairIndices);
        if(it != collisions.end())
        {
          it->second.normal   = normal;
          it->second.point    = point;
          it->second.active   = 1;
        }else
        {
          Collision &col = collisions[pairIndices];
          col.normal  = normal;
          col.point   = point;
          col.indices = pairIndices;
          col.active  = 1;
        }
      }
    }
  }
}

void SolveCollisions(int iterationsCount)
{
  for(int i = 0; i < iterationsCount; i++)
  {
    for (CollisionMap::iterator it = collisions.begin(); it != collisions.end(); it++)
    {
      it->second.Solve();
    }
  }
}

void display()
{
  float timeStep = 1e-2f;
  for(size_t i = 0; i < spheres.size(); i++)
  {
    if(spheres[i].isStatic)
      spheres[i].acceleration = vec2(0, 0);
    else
      spheres[i].acceleration = vec2(0, -100.0);

    spheres[i].IntegrateVelocity(timeStep);
  }
  
  CollisionMap::iterator it;
  for (it = collisions.begin(); it != collisions.end(); it++)
  {
    it->second.active = 0;
  }

  FindCollisions();

  it = collisions.begin();
  while(it != collisions.end())
  {
    if (!it->second.active)
    {
      collisions.erase(it++);
    }
    else
    {
      it->second.PreStep();
      ++it;
    }
  }

  SolveCollisions(10);

  for(size_t i = 0; i < spheres.size(); i++)
  {
    spheres[i].IntegratePosition(timeStep);
  }

  glClear(GL_COLOR_BUFFER_BIT);

  glColor3f(0,0,0);

  for(size_t i = 0; i < spheres.size(); i++)
  {
    spheres[i].pos.draw_circle(spheres[i].radius);
  }

  glColor3f(1,0,0);
  for (it = collisions.begin(); it != collisions.end(); it++)
  {
    it->second.point.draw_circle(0.1);
  }

  glFlush();
}



void ReSize(GLsizei width ,GLsizei height)
{
	glViewport(0,0,width,height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glOrtho(-40,40,-40,40,-10,10);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void  keyboard(unsigned char key,int x,int y)
{
		switch(key)
		{
		  case 'w':     break;
		  case 'a':     break;
		}
}


void motion(int x,int y)
{
}

void timer(int = 0)
{
	display();
	glutTimerFunc(20,timer,0);
}


int main(int argc, char* argv[])
{
  Sphere newbie;

  newbie.velocity = vec2(0, 0);

  newbie.invMass = 0;
  newbie.isStatic = 1;
  newbie.radius = 50;

  newbie.pos = vec2(-40.0f, -50.0f);
  spheres.push_back(newbie);

  newbie.pos = vec2( 40.0f, -50.0f);
  spheres.push_back(newbie);

  newbie.pos = vec2(-70.0f, 20.0f);
  spheres.push_back(newbie);

  newbie.pos = vec2( 70.0f, 20.0f);
  spheres.push_back(newbie);

  newbie.radius = 1.0f;
  newbie.invMass = 1.0f;
  newbie.isStatic = 0;

  for(int i = 0; i < 200; i++)
  {
    newbie.pos = vec2(0 + (i % 2) * 2.0f, 5 + 1.5f * i);
    spheres.push_back(newbie);
  }

  glutInit(&argc,argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(500,500);
  glutInitWindowPosition(400,400);
  glutCreateWindow("test-physx2");
  glClearColor(1,1,1,1);
  ReSize(500,500);
  glutMotionFunc(motion);
  glutKeyboardFunc(keyboard);
  glutDisplayFunc(display);
  timer();


  glutMainLoop();
  return 0;
}
