#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <bits/stdc++.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
double ar[8][4],br[8][4],t=10,ar1[4][3],PI=3.1415926535897932384626433832;
double cos_an[15],sin_an[15],rx[4][4],ry[4][4],rz[4][4];
double xp,yp,zp,q,dx,dy,dz,cop_x=-3,cop_y=-3,cop_z=-4;

void midpoint1(double x1, double y1, double x2, double y2) {

    int flag = 0, incrY;
    int dx = abs(x1-x2), dy = abs(y1-y2);

	if(dy>dx) {
		swap(x1, y1);
		swap(x2, y2);
		swap(dx, dy);
		flag = 1;
	}

	if(x1>x2){
		swap(x1, x2);
		swap(y1, y2);
	}

	if(y1>y2) {  incrY =- 1;}
	else {  incrY = 1;}

    double d = 2*dy - dx;
    double incrE = 2*dy;
    double incrNE = 2*(dy-dx);

    while(x1<=x2) {

        if(flag==1) {
            glBegin(GL_POINTS);

            glColor3f(0.0f,1.0f,0.0f);
                glVertex2f(y1,x1);

        glEnd();
        }else {
            glBegin(GL_POINTS);

            glColor3f(0.0f,1.0f,0.0f);
                glVertex2f(x1,y1);

        glEnd();
        }

        if(d<=0) {  d += incrE;}
        else{
            d += incrNE;
            y1 += incrY;
        }
        x1++;
    }
}
void ini(){
    rx[0][0]=1,rx[0][1]=0,rx[0][2]=0,rx[0][3]=0;
    rx[0][0]=0,rx[0][1]=cos(t*PI/180),rx[0][2]=sin(t*PI/180),rx[0][3]=0;
    rx[2][0]=0,rx[2][1]=sin(t*PI/180),rx[2][2]=cos(t*PI/180),rx[0][3]=0;
    rx[3][0]=0,rx[3][1]=0,rx[3][2]=0,rx[3][3]=1;
    ry[0][0]=cos(t*PI/180),ry[0][1]=0,ry[0][2]=sin(t*PI/180),ry[0][3]=0;
    ry[1][0]=0,ry[1][1]=1,ry[1][2]=0,ry[1][3]=0;
    ry[2][0]=-sin(t*PI/180),ry[2][1]=0,ry[2][2]=cos(t*PI/180),ry[1][3]=0;
    ry[3][0]=0,ry[3][1]=0,ry[3][2]=0,ry[3][3]=1;
    rz[0][0]=cos(t*PI/180),rz[0][1]=-sin(t*PI/180);rz[0][2]=0,rz[0][3]=0;
    rz[1][0]=0,rz[1][1]=1;rz[1][2]=0,rz[1][3]=0;
    rz[2][0]=0,rz[2][1]=0;rz[2][2]=1,rz[2][3]=0;
    rz[3][0]=0,rz[3][1]=0;rz[3][2]=0,rz[3][3]=1;
    ar[0][0]=70,ar[0][1]=10,ar[0][2]=-2,ar[0][3]=1;
    ar[1][0]=120,ar[1][1]=60,ar[1][2]=-2,ar[1][3]=1;
    ar[2][0]=70,ar[2][1]=110,ar[2][2]=-2,ar[2][3]=1;
    ar[3][0]=20,ar[3][1]=60,ar[3][2]=-2,ar[3][3]=1;
    ar[4][0]=40,ar[4][1]=10,ar[4][2]=-4,ar[4][3]=1;
    ar[5][0]=90,ar[5][1]=60,ar[5][2]=-4,ar[5][3]=1;
    ar[6][0]=40,ar[6][1]=110,ar[6][2]=-4,ar[6][3]=1;
    ar[7][0]=-10,ar[7][1]=60,ar[7][2]=-4,ar[7][3]=1;
printf("first  %lf %lf %lf\n %lf %lf %lf %lf %lf\n",ar[7][0],ar[7][1],ar[6][0],ar[6][1],ar[5][0],ar[5][1],ar[4][0],ar[4][1]);
}


void d(){
    glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
      glVertex2f(70,10 );
      glVertex2f( 120,60 );
      glVertex2f(70,10 );
      glVertex2f( 20,60 );
      glVertex2f(20,60);
      glVertex2f( 70,110 );
      glVertex2f(70,110);
      glVertex2f( 120,60);
    glEnd();

}
void cube_x(){


    glFlush();
    for(int i=0;i<1000;i++);
    for(int i=0;i<=7;i++){
            double v=t*PI/180;
        ar1[i][0]=ar[i][0];
        ar1[i][1]=-sin(t*PI/180)*ar[i][2]+cos(t*PI/180)*ar[i][1];
        ar1[i][2]=sin(t*PI/180)*ar[i][1]+cos(t*PI/180)*ar[i][2];

        //printf("%lf %lf %lf\n",ar1[i][0],ar1[i][1],PI);
    }
   // printf("next  %lf %lf %lf\n %lf %lf %lf %lf %lf\n",ar1[7][0],ar1[7][1],ar1[6][0],ar1[6][1],ar1[5][0],ar1[5][1],ar1[4][0],ar1[4][1]);
    q=sqrt(cop_x*cop_x+cop_y*cop_y+cop_z*cop_x);
dx=cop_x/q;
dy=cop_y/q;
dz=(cop_z-zp)/q;
for(int i=0;i<=7;i++){
    br[i][0]=(ar1[i][0]-ar1[i][2]*dx/dz+zp*dx/dz);
    br[i][1]=(ar1[i][1]-ar1[i][2]*dy/dz+zp*dy/dz);
}
    cout<<br[1][0]<<' '<<br[1][1]<<endl;
    /*glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);

     glVertex2f(br[0][0],br[0][1] );
      glVertex2f( br[1][0],br[1][1] );
     glColor3f(0.0f, 0.0f, 1.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[1][0],br[1][1] );

     glColor3f(1.0f, 1.0f, 0.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[3][0],br[3][1] );


     glColor3f(0.0f, 1.0f, 0.0f);
      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[3][0],br[3][1] );


      glVertex2f(br[4][0],br[4][1] );
      glVertex2f(br[5][0],br[5][1] );

      glVertex2f(br[5][0],br[5][1] );
      glVertex2f(br[6][0],br[6][1] );

      glVertex2f(br[6][0],br[6][1] );
      glVertex2f(br[7][0],br[7][1] );

      glVertex2f(br[7][0],br[7][1] );
      glVertex2f( br[4][0],br[4][1] );

      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[4][0],br[4][1] );

      glVertex2f(br[5][0],br[5][1] );
      glVertex2f(br[1][0],br[1][1] );

      glVertex2f(br[2][0],br[2][1] );
      glVertex2f(br[6][0],br[6][1] );

      glVertex2f(br[7][0],br[7][1] );
      glVertex2f(br[3][0],br[3][1] );

        glEnd();*/
        midpoint1(br[6][0],br[6][1],br[5][0],br[5][1]);
        midpoint1(br[4][0],br[4][1],br[5][0],br[5][1]);
        midpoint1(br[0][0],br[0][1],br[3][0],br[3][1]);
        midpoint1(br[2][0],br[2][1],br[3][0],br[3][1]);
        midpoint1(br[2][0],br[2][1],br[1][0],br[1][1]);
        midpoint1(br[0][0],br[0][1],br[1][0],br[1][1]);
        midpoint1(br[7][0],br[7][1],br[3][0],br[3][1]);
        midpoint1(br[2][0],br[2][1],br[6][0],br[6][1]);
        midpoint1(br[1][0],br[1][1],br[5][0],br[5][1]);
        midpoint1(br[4][0],br[4][1],br[7][0],br[7][1]);
        midpoint1(br[4][0],br[4][1],br[0][0],br[0][1]);
        midpoint1(br[6][0],br[6][1],br[7][0],br[7][1]);
}
void cube_y(){


    glFlush();
    for(int i=0;i<1000;i++);
    for(int i=0;i<=7;i++){
            double v=t*PI/180;
        ar1[i][0]=sin(t*PI/180)*ar[i][2]+cos(t*PI/180)*ar[i][0];
        ar1[i][1]=ar[i][1];
        ar1[i][2]=-sin(t*PI/180)*ar[i][0]+cos(t*PI/180)*ar[i][2];

        //printf("%lf %lf %lf\n",ar1[i][0],ar1[i][1],PI);
    }
    printf("next  %lf %lf %lf\n %lf %lf %lf %lf %lf\n",ar1[7][0],ar1[7][1],ar1[6][0],ar1[6][1],ar1[5][0],ar1[5][1],ar1[4][0],ar1[4][1]);

   q=sqrt(cop_x*cop_x+cop_y*cop_y+cop_z*cop_x);
dx=cop_x/q;
dy=cop_y/q;
dz=(cop_z-zp)/q;
for(int i=0;i<=7;i++){
    br[i][0]=(ar1[i][0]-ar1[i][2]*dx/dz+zp*dx/dz);
    br[i][1]=(ar1[i][1]-ar1[i][2]*dy/dz+zp*dy/dz);
}
    cout<<br[1][0]<<' '<<br[1][1]<<endl;
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
     glVertex2f(br[0][0],br[0][1] );
      glVertex2f( br[1][0],br[1][1] );
     glColor3f(0.0f, 0.0f, 1.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[1][0],br[1][1] );
     glColor3f(1.0f, 1.0f, 0.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[3][0],br[3][1] );
     glColor3f(0.0f, 1.0f, 0.0f);
      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[3][0],br[3][1] );

      glVertex2f(br[4][0],br[4][1] );
      glVertex2f(br[5][0],br[5][1] );

      glVertex2f(br[5][0],br[5][1] );
      glVertex2f(br[6][0],br[6][1] );

      glVertex2f(br[6][0],br[6][1] );
      glVertex2f(br[7][0],br[7][1] );

      glVertex2f(br[7][0],br[7][1] );
      glVertex2f( br[4][0],br[4][1] );

      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[4][0],br[4][1] );
      //glVertex2f(ar[5][0],br[5][1] );
      //glVertex2f(br[1][0],br[1][1] );
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f(br[6][0],br[6][1] );
      glVertex2f(br[7][0],br[7][1] );
      glVertex2f(br[3][0],br[3][1] );
        glEnd();

}
void cube_xy(){


    glFlush();
    for(int i=0;i<1000;i++);
    for(int i=0;i<=7;i++){
            double v=t*PI/180;
        ar1[i][0]=sin(t*PI/180)*ar[i][2]+cos(t*PI/180)*ar[i][0];
        ar1[i][1]=ar[i][0]*sin(t*PI/180)*sin(t*PI/180)+ar[i][1]*cos(t*PI/180)-cos(t*PI/180)*sin(t*PI/180)*ar[i][2];
        ar1[i][2]=-sin(t*PI/180)*cos(t*PI/180)*ar[i][0]+sin(t*PI/180)*ar[i][1]+cos(t*PI/180)*cos(t*PI/180)*ar[i][2];

        //printf("%lf %lf %lf\n",ar1[i][0],ar1[i][1],PI);
    }
    q=sqrt(cop_x*cop_x+cop_y*cop_y+cop_z*cop_x);
dx=cop_x/q;
dy=cop_y/q;
dz=(cop_z-zp)/q;
for(int i=0;i<=7;i++){
    br[i][0]=(ar1[i][0]-ar1[i][2]*dx/dz+zp*dx/dz);
    br[i][1]=(ar1[i][1]-ar1[i][2]*dy/dz+zp*dy/dz);
}
    cout<<br[1][0]<<' '<<br[1][1]<<endl;
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
     glVertex2f(br[0][0],br[0][1] );
      glVertex2f( br[1][0],br[1][1] );
     glColor3f(0.0f, 0.0f, 1.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[1][0],br[1][1] );
     glColor3f(1.0f, 1.0f, 0.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[3][0],br[3][1] );
     glColor3f(0.0f, 1.0f, 0.0f);
      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[3][0],br[3][1] );

      glVertex2f(br[4][0],br[4][1] );
      glVertex2f(br[5][0],br[5][1] );

      glVertex2f(br[5][0],br[5][1] );
      glVertex2f(br[6][0],br[6][1] );

      glVertex2f(br[6][0],br[6][1] );
      glVertex2f(br[7][0],br[7][1] );

      glVertex2f(br[7][0],br[7][1] );
      glVertex2f( br[4][0],br[4][1] );

      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[4][0],br[4][1] );

      glVertex2f(br[5][0],br[5][1] );
      glVertex2f(br[1][0],br[1][1] );

      glVertex2f(br[2][0],br[2][1] );
      glVertex2f(br[6][0],br[6][1] );

      glVertex2f(br[7][0],br[7][1] );
      glVertex2f(br[3][0],br[3][1] );
        glEnd();

}
void cube_z(){


    glFlush();
    for(int i=0;i<1000;i++);
    for(int i=0;i<=7;i++){
            double v=t*PI/180;
        ar1[i][0]=-sin(t*PI/180)*ar[i][1]+cos(t*PI/180)*ar[i][0];
        ar1[i][1]=sin(t*PI/180)*ar[i][0]+cos(t*PI/180)*ar[i][1];
        ar1[i][2]=ar[i][2];

        //printf("%lf %lf %lf\n",ar1[i][0],ar1[i][1],PI);
    }
   q=sqrt(cop_x*cop_x+cop_y*cop_y+cop_z*cop_x);
dx=cop_x/q;
dy=cop_y/q;
dz=(cop_z-zp)/q;
for(int i=0;i<=7;i++){
    br[i][0]=(ar1[i][0]-ar1[i][2]*dx/dz+zp*dx/dz);
    br[i][1]=(ar1[i][1]-ar1[i][2]*dy/dz+zp*dy/dz);
}
/*
    cout<<br[1][0]<<' '<<br[1][1]<<endl;
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.0f, 0.0f);
     glVertex2f(br[0][0],br[0][1] );
      glVertex2f( br[1][0],br[1][1] );
     glColor3f(0.0f, 0.0f, 1.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[1][0],br[1][1] );
     glColor3f(1.0f, 1.0f, 0.0f);
      glVertex2f(br[2][0],br[2][1] );
      glVertex2f( br[3][0],br[3][1] );
     glColor3f(0.0f, 1.0f, 0.0f);
      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[3][0],br[3][1] );

      glVertex2f(br[4][0],br[4][1] );
      glVertex2f(br[5][0],br[5][1] );

      glVertex2f(br[5][0],br[5][1] );
      glVertex2f(br[6][0],br[6][1] );

      glVertex2f(br[6][0],br[6][1] );
      glVertex2f(br[7][0],br[7][1] );

      glVertex2f(br[7][0],br[7][1] );
      glVertex2f( br[4][0],br[4][1] );

      glVertex2f(br[0][0],br[0][1] );
      glVertex2f(br[4][0],br[4][1] );

      glVertex2f(br[5][0],br[5][1] );
      glVertex2f(br[1][0],br[1][1] );

      glVertex2f(br[2][0],br[2][1] );
      glVertex2f(br[6][0],br[6][1] );

      glVertex2f(br[7][0],br[7][1] );
      glVertex2f(br[3][0],br[3][1] );
        glEnd();*/
        midpoint1(br[6][0],br[6][1],br[5][0],br[5][1]);
        midpoint1(br[4][0],br[4][1],br[5][0],br[5][1]);
        midpoint1(br[0][0],br[0][1],br[3][0],br[3][1]);
        midpoint1(br[2][0],br[2][1],br[3][0],br[3][1]);
        midpoint1(br[2][0],br[2][1],br[1][0],br[1][1]);
        midpoint1(br[0][0],br[0][1],br[1][0],br[1][1]);
        midpoint1(br[7][0],br[7][1],br[3][0],br[3][1]);
        midpoint1(br[2][0],br[2][1],br[6][0],br[6][1]);
        midpoint1(br[1][0],br[1][1],br[5][0],br[5][1]);
        midpoint1(br[4][0],br[4][1],br[7][0],br[7][1]);
        midpoint1(br[4][0],br[4][1],br[0][0],br[0][1]);
        midpoint1(br[6][0],br[6][1],br[7][0],br[7][1]);
}
void display(){
    ini();
   while(1){
    glClear(GL_COLOR_BUFFER_BIT);


    cube_x();
    //d();
    glFlush();
    t+=5;
    Sleep(100);
    glutSwapBuffers();
    }
}
int main(int argc, char *argv[]){
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE);
    glutInitWindowSize(640,640);
    glutInitWindowPosition(0,0);
    glutCreateWindow("rotating cube");
    glClearColor(0.0,0.0,0.0,.70);


	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-150.0, 150.0, -150.0, 150.0);

    glutDisplayFunc(display);

    glutMainLoop();

}
