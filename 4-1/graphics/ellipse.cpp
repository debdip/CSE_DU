#include <windows.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace std;
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

            glColor3f(1.0f,0.0f,0.50f);
                glVertex2f(y1,x1);

        glEnd();
        }else {
            glBegin(GL_POINTS);

            glColor3f(1.0f,0.0f,0.50f);
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
            glBegin(GL_POINTS);

            glColor3f(1.0f,1.0f,.3f);
                glVertex2f(y1,x1);

        glEnd();
        }else {
            glBegin(GL_POINTS);

            glColor3f(1.0f,1.0f,.3f);
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

void ellipse(double x1,double y1,double a,double b){
    double d2,x,y;
     x=0;
     y=b;
    double dl=b*b-(a*a*b)+(.25*a*a);
    //double x1=30,y1=-10;
    midpoint1(x1,y1,x1+x,y1+y);
    while((a*a*(y-0.5))>(b*b*(x+1))){
        if(dl<0)
            dl+=b*b*(2*x+3);
        else{
            dl+=b*b*(2*x+3)+a*a*(-2*y+2);
            y--;
        }
        x++;
        Sleep(100);
        glFlush();
        glPointSize(3.0);
        midpoint1(x1,y1,x1+x,y1+y);
        midpoint1(x1,y1,x1-x,y1+y);
        midpoint1(x1,y1,x1-x,y1-y);
        midpoint1(x1,y1,x1+x,y1-y);
    }
    d2=b*b*(x+0.5)*(x+.5)+a*a*(y-1)*(y-1)-a*a*b*b;
    while(y>0){
        if(d2<0){
            d2+=b*b*(2*x+2)+a*a*(-2*y+3);
            x++;
        }
        else{
            d2+=a*a*(-2*y+3);
        }
        y--;
        Sleep(100);
        glFlush();
        glPointSize(3.0);
        midpoint2(x1,y1,x1+x,y1+y);
        midpoint2(x1,y1,x1-x,y1+y);
        midpoint2(x1,y1,x1-x,y1-y);
        midpoint2(x1,y1,x1+x,y1-y);

    }


}
void check(){
    double x=40,y=120;
   //while(x<=100){
           // for(int i=0;i<=1000000;i++);
             Sleep(230);
        glClear(GL_COLOR_BUFFER_BIT);
        glBegin(GL_POINTS);
        for(int i=-150;i<150;i++)
            glVertex2f(i,0);
         for(int i=-150;i<150;i++)
            glVertex2f(0,i);
            glEnd();
        ellipse(-50,0,140,60);
        x+=5;y-=5;
        glFlush();
        printf("What\n");
   // ellipse(100.0,60.0);
    //}
}
void display(){

   //while(1){
    glClear(GL_COLOR_BUFFER_BIT);
    //double x1=20.0,y1=10.0,y2=50.0,x2=0.0;

    check();
    glFlush();
   //}
}
int main(int argc, char *argv[]){
    glutInit(&argc, argv);
    glutInitWindowSize(640,640);
    glutInitWindowPosition(0,0);
    glutCreateWindow("Circle Drawing");
    glClearColor(0.0,0.0,0.0,.0);


	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-150.0, 150.0, -150.0, 150.0);

    glutDisplayFunc(display);

    glutMainLoop();

}
