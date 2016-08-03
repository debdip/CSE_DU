#include <windows.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
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
