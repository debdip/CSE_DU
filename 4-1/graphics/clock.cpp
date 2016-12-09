#include <windows.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
//#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//using namespace std;
void midpoint1(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);
            glColor3f(.40f, .10f, 0.60f);
                glVertex2f(y1, x1);
        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }


}
void midpoint2(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);

            glColor3f(.6f,.4f,.3f);
                glVertex2f(x1,y1);

        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }

}
void midpoint3(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);

            glColor3f(1.0f, 1.0f, 0.0f);
                glVertex2f(x1, -y1);

        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }

}
void midpoint4(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);

            glColor3f(.50f, .50f, 0.50f);
                glVertex2f(y1, -x1);
        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }

}
void midpoint5(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);
            glColor3f(.90f, .90f, 0.90f);
                glVertex2f(-y1, -x1);
        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }

}
void midpoint6(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);
            glColor3f(1.0f, 1.0f, 0.0f);
                glVertex2f(-x1, -y1);

        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }

}
void midpoint7(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);

            glColor3f(1.0f, .20f, 0.50f);
                glVertex2f(-x1, y1);

        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }

}
void midpoint8(double x1,double y1,double x2,double y2){
    double dx=x2-x1;
    double dy=y2-y1;
    double d=2*dy-dx;
    double incrE=2*dy;
    double incrNE=2*(dy-dx);
    while(x1<=x2){
        glBegin(GL_POINTS);

            glColor3f(.40f, .40f, .40f);
                glVertex2f(-y1, x1);

        glEnd();
        if(d<=0){
            d+=incrE;
            x1+=.05;
        }else{
            d+=incrNE;
            x1+=.05;
            y1+=.05;
        }
    }

}
void circle(double x1,double y1,double r){

    x1=0,y1=r;

    double d=5/4-r;
     while(y1>=x1){
            glFlush();
  //
        glClear(GL_COLOR_BUFFER_BIT);
        for(int i=0;i<=10000000;i++);
        midpoint1(0,0,y1,x1);



        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1-=.5;
        }
        x1+=.5;
    }
     printf("%f %f\n",x1,y1);
    //x1=0,y1=r;

     d=1-r;
     //glClear(GL_COLOR_BUFFER_BIT);
     while(x1>=0){

        glFlush();
        for(int i=0;i<10000000;i++);
        glClear(GL_COLOR_BUFFER_BIT);

        midpoint2(0,0,y1,x1);
        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1+=.5;
        }
        x1-=.5;
     }
 printf("%f %f\n",x1,y1);
x1=0,y1=r;

     d=1-r;
     while(y1>=x1){
        glFlush();
        for(int i=0;i<10000000;i++);
        glClear(GL_COLOR_BUFFER_BIT);
        midpoint3(0,0,y1,x1);
        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1-=.5;
        }
        x1+=.5;
     }
      printf("%f %f\n",x1,y1);
     //x1=0,y1=r;
     d=1-r;
     while(x1>=0){
        glFlush();
        for(int i=0;i<10000000;i++);
        glClear(GL_COLOR_BUFFER_BIT);
        midpoint4(0,0,y1,x1);
        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1+=.5;
        }
        x1-=.5;
     }
      printf("%f %f\n",x1,y1);
     x1=0,y1=r;

     d=1-r;
     while(y1>=x1){
        glFlush();
        for(int i=0;i<10000000;i++);
        glClear(GL_COLOR_BUFFER_BIT);
        midpoint5(0,0,y1,x1);
        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1-=.5;
        }
        x1+=.5;
     }
      printf("%f %f\n",x1,y1);
     //x1=0,y1=r;

     d=1-r;
     while(x1>=0){
            printf("working\n");
        glFlush();
        for(int i=0;i<10000000;i++);
        glClear(GL_COLOR_BUFFER_BIT);
        midpoint6(0,0,y1,x1);
        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1+=.5;
        }
        x1-=.5;
     }


      printf("%f %f\n",x1,y1);
     x1=0,y1=r;

     d=1-r;
     while(y1>=x1){
        glFlush();
        for(int i=0;i<10000000;i++);
        glClear(GL_COLOR_BUFFER_BIT);
        midpoint7(0,0,y1,x1);
        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1-=.5;
        }
        x1+=.5;
     }
      printf("%f %f\n",x1,y1);

     //x1=0,y1=r;

     d=1-r;
     while(x1>=0){
        glFlush();
        for(int i=0;i<10000000;i++);
        glClear(GL_COLOR_BUFFER_BIT);
        midpoint8(0,0,y1,x1);
        if(d<0)
            d+=2*x1+3;
        else{
            d+=2*(x1-y1)+5;
            y1+=.5;
        }
        x1-=.5;
     }


}
void display(){

    glClear(GL_COLOR_BUFFER_BIT);
    //double x1=20.0,y1=10.0,y2=50.0,x2=0.0;
    circle(0.0,0.0,80);
    glFlush();
}
int main(int argc, char *argv[]){
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
