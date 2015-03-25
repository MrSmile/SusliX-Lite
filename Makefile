
HEADER = AABB2.h Collider.h Collision.h Coords2.h Geom.h Joints.h Manifold.h PhysSystem.h RigidBody.h Solver.h Vector2.h
SOURCE = main.cpp FastSolver.cpp
FLAGS = -std=c++11
LIBS = -lsfml-window -lsfml-system -lsfml-graphics
PROGRAM = SusliX-Lite


debug: $(SOURCE) $(HEADER)
	g++ -g -O0 -DDEBUG $(FLAGS) $(SOURCE) $(LIBS) -o $(PROGRAM)

release: $(SOURCE) $(HEADER)
	g++ -Ofast -flto -mtune=native $(FLAGS) $(SOURCE) $(LIBS) -o $(PROGRAM)

clean:
	rm $(PROGRAM)
