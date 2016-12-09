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
double rect_xy[4][2],rect_z[4],col_red[4],line1_x[200],line1_y[200],line2_x[200],line2_y[200],gl_color;
double rotat1_x[200],rotat1_y[200],rotat2_x[200],rotat2_y[200];
int fl1=0,fl2=0;
double min_x,max_y,max_x,min_y;
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


                line1_x[fl1]=y1;
                line1_y[fl1]=x1;



        }else {

               line1_x[fl1]=x1;
                line1_y[fl1]=y1;


        }

        if(d<=0) {  d += incrE;}
        else{
            d += incrNE;
            y1 += incrY;
        }
        fl1++;
        x1++;
    }
}
void midpoint2(double x1, double y1, double x2, double y2) {

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

                line2_x[fl2]=y1;
                line2_y[fl2]=x1;

        }else {

                line2_x[fl2]=x1;
                line2_y[fl2]=y1;

        }

        if(d<=0) {  d += incrE;}
        else{
            d += incrNE;
            y1 += incrY;
        }
        fl2++;
        x1++;

    }
}
void midpoint3(double x1, double y1, double x2, double y2) {
    double l=1;
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

        if(((min_x<=y1)&&(y1<=max_x))&&((min_y<=x1)&&(x1<=max_y))){
            glColor3f(1.0f,l/100,gl_color/100);
                glVertex2f(y1,x1);
            }

        glEnd();
        }else {
            glBegin(GL_POINTS);

            if(((min_x<=x1)&&(x1<=max_x))&&((min_y<=y1)&&(y1<=max_y))){
            glColor3f(1.0f,l/100,gl_color/100);
                glVertex2f(x1,y1);
            }

        glEnd();
        }

        if(d<=0) {  d += incrE;}
        else{
            d += incrNE;
            y1 += incrY;
        }
        //printf("this is %lf %lf color---%lf %lf \n",x1,y1,l/100,gl_color);
        l++;
        x1++;
    }
    printf("what\n");
}
 void ini(){
    rect_xy[0][0]=-20;
    rect_xy[0][1]=120;
    rect_xy[1][0]=80;
    rect_xy[1][1]=120;
    rect_xy[2][0]=-20;
    rect_xy[2][1]=20;
    rect_xy[3][0]=80;
    rect_xy[3][1]=120;
    midpoint1(-20,120,-20,20);
    midpoint2(80,120,80,20);
    //glClear(GL_COLOR_BUFFER_BIT);

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

void cube_y(){


    glFlush();
    //for(int i=0;i<1000;i++);
    for(int i=0;i<100;i++){
            //double v=t*PI/180;
       // ar1[i][0]=sin(t*PI/180)*ar[i][2]+cos(t*PI/180)*ar[i][0];
        //ar1[i][1]=ar[i][1];
       // ar1[i][2]=-sin(t*PI/180)*ar[i][0]+cos(t*PI/180)*ar[i][2];
        rotat1_x[i]=sin(t*PI/180)*(-3)+cos(t*PI/180)*line1_x[i];
        rotat1_y[i]=line1_y[i];
        rotat2_x[i]=sin(t*PI/180)*(-3)+cos(t*PI/180)*line2_x[i];
        rotat2_y[i]=line2_y[i];
        gl_color=i;
        midpoint3(rotat1_x[i],rotat1_y[i],rotat2_x[i],rotat2_y[i]);
        printf(" why %lf %lf %lf %lf\n",rotat1_x[i],rotat2_x[i],rotat1_y[i],rotat2_y[i]);
    }
    //printf("next  %lf %lf %lf\n %lf %lf %lf %lf %lf\n",ar1[7][0],ar1[7][1],ar1[6][0],ar1[6][1],ar1[5][0],ar1[5][1],ar1[4][0],ar1[4][1]);


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

void display(){
    ini();
   while(1){
    glClear(GL_COLOR_BUFFER_BIT);
     glFlush();
    //ini();
    cube_y();
    //d();

    t+=5;
    Sleep(100);
    glutSwapBuffers();
    }
}
int main(int argc, char *argv[]){

    printf("Enter clipping points\n");
    cin>>min_x>>max_y>>max_x>>min_y;
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
