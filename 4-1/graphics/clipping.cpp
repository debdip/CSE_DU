#include <windows.h>

#include <iostream>
#include <GL/glut.h>

#include <stdlib.h>
using namespace std;
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
            glBegin(GL_POINTS);

            glColor3f(1.0f,1.0f,0.0f);
                glVertex2f(y1,x1);

        glEnd();
        }else {
            glBegin(GL_POINTS);

            glColor3f(1.0f,1.0f,0.0f);
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
void clipping(double x1, double y1, double x2, double y2) {

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
            glColor3f(1.0f,1.0f,0.0f);
                glVertex2f(y1,x1);
            }
            else{
                glColor3f(1.0f,0.0f,0.0f);
                glVertex2f(y1,x1);
            }

        glEnd();
        }else {
            glBegin(GL_POINTS);
            if(((min_x<=x1)&&(x1<=max_x))&&((min_y<=y1)&&(y1<=max_y))){
            glColor3f(1.0f,1.0f,0.0f);
                glVertex2f(x1,y1);
            }
            else{
                glColor3f(1.0f,0.0f,0.0f);
                glVertex2f(x1,y1);
            }

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
void rectangle(){
    midpoint1(min_x,max_y,max_x,max_y);
    midpoint1(min_x,max_y,min_x,min_y);
    midpoint1(max_x,max_y,max_x,min_y);
    midpoint1(min_x,min_y,max_x,min_y);


}
void clip(){


    rectangle();
    clipping(-120,-50,-70,100);  //sample points
    clipping(-70,100,-20,-40);  //sample points
    clipping(-20,-40,40,120);   //sample points
    clipping(40,120,80,-40);   //sample points
    clipping(80,-40,120,120);  //sample points

}
void display(){

    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(2.0);
    //double x1=20.0,y1=10.0,y2=50.0,x2=0.0;
    clip();
    glFlush();
}
int main(int argc, char *argv[]){
    cout<<"give rectangle one diagonal two points"<<endl;
    cin>>min_x>>max_y>>max_x>>min_y;
    glutInit(&argc, argv);
    glutInitWindowSize(640,640);
    glutInitWindowPosition(0,0);
    glutCreateWindow("Circle Drawing");
    glClearColor(0.70,0.70,0.50,.70);


	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(-150.0, 150.0, -150.0, 150.0);

    glutDisplayFunc(display);

    glutMainLoop();

}
