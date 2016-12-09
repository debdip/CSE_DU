

#include <windows.h>


#include <GL/glut.h>

#include<stdio.h>
#include <stdlib.h>

int x0,y0,x1,y1,quardant,xmin=-200,xmax=200,ymin=-70,ymax=70,a,b,c,d,a1,b1,c1,d1,top=1,bottom=2,left=8,right=4,flag=0;
float Color1=1.0, Color2=1.0, Color3=1.0;
void Init(int w, int h)
{
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glViewport(0,0, (GLsizei)w,(GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(w/-2,w/2,h/-2,h/2);
}
void drawpixel(int x,int y)
{
    glBegin(GL_POINTS);
    glVertex2f(x,y);
    glColor3f(Color1,Color2,Color3);
    glEnd();
    //glFlush();
}
void pixel_position(int x,int y)
{   if(quardant == 1)
        drawpixel(x,y);
    else if(quardant == 2)
        drawpixel(y,x);
    else if(quardant == 3)
        drawpixel(y,-x);
    else if(quardant == 4)
        drawpixel(-x,y);
    else if(quardant == 5)
        drawpixel(-x,-y);
    else if(quardant == 6)
        drawpixel(-y,-x);
    else if(quardant == 7)
        drawpixel(-y,x);
    else
        drawpixel(x,-y);
}
void calculate_quardant()
{   int dx,dy;
    double m;
    dx=x1-x0;
    dy=y1-y0;
    m=(double)dy/dx;
    if(m>=0 && m<=1)
    {
        if(x1>=x0) quardant=1;
        else quardant=5;

    }
    else if(m>1)
    {
         if(x1>=x0) quardant=2;
        else  quardant=6;

    }
    else if(m<-1)
    {
         if(x1>=x0) quardant=3;
        else  quardant=7;
    }
    else
    {
        if(x1<=x0) quardant=4;
        else  quardant=8;
    }
}
void midpointline(int x0,int y0,int x1,int y1)
{
    int d,dx,dy,inE,inNE,x,y;
    dx = x1-x0;
    dy = y1-y0;
    d = 2*dy-dx;
    inE = 2*dy;
    inNE = 2*(dy-dx);
    x = x0;
    y = y0;
    //printf("%d %d %d %d\n",x0,y0,x1,y1);
    pixel_position(x,y);
    while(x<x1){
        if(d<=0){
            d = d+inE;
            x++;
        }
        else{
            d=d+inNE;
            x++;
            y++;
        }
        pixel_position(x,y);
    }
}
void line()
{
    calculate_quardant();
    if(quardant == 1)
        midpointline(x0,y0,x1,y1);
    else if(quardant == 2)
        midpointline(y0,x0,y1,x1);
    else if(quardant == 3)
        midpointline(-y0,x0,-y1,x1);
    else if(quardant == 4)
        midpointline(-x0,y0,-x1,y1);
    else if(quardant == 5)
        midpointline(-x0,-y0,-x1,-y1);
    else if(quardant == 6)
        midpointline(-y0,-x0,-y1,-x1);
    else if(quardant == 7)
        midpointline(y0,-x0,y1,-x1);
    else
        midpointline(x0,-y0,x1,-y1);
    //glFlush();
}
int compute_code(int x,int y)
{
    int code=0;
    if(y>ymax) code|=top;
    else if(y<ymin) code|=bottom;
    else if(x>xmax) code|=right;
    else if(x<xmin) code|=left;
    return code;
}
void cohen_sutherland()
{
    int code,code0,code1;
    code0=compute_code(a,b);
    code1=compute_code(c,d);
    while(1)
    {   double x,y;
        if(!(code0|code1)) { flag=1; break;}
        else if(code0&code1) { break;}
        else
        {
            code=code0?code0:code1;
            if(code& top)
            {
                x=a+(c-a)*(ymax-b)/(d-b);y=ymax;
            }
            else if(code &bottom)
            {
                 x=a+(c-a)*(ymin-b)/(d-b);y=ymin;
            }
            else if(code&right)
            {
                 y=b+(d-b)*(xmax-a)/(c-a);x=xmax;
            }
            else
            {
                 y=b+(d-b)*(xmin-a)/(c-a);x=xmin;
            }
            if(code==code0)
            {
                a=x,b=y,code0=compute_code(a,b);
            }
            else
            {
                 c=x,d=y,code1=compute_code(c,d);
            }
        }

    }

}
static void display(void)
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glColor4f(0,0,0,1);
    glPointSize(3);
    glBegin(GL_POINTS);int i;
    for(i=-400;i<=400;i++)
        glVertex2f(i,0);
    for(i=-300;i<=300;i++)
        glVertex2f(0,i);
    glEnd();
    x0=xmax,y0=ymax,x1=x0,y1=ymin,Color1=1,Color2=0,Color3=0;
    line();
    x0=xmax,y0=ymax,x1=xmin,y1=ymax;
    line();
    x0=xmin,y0=ymax,x1=xmin,y1=ymin;
    line();
    x0=xmin,y0=ymin,x1=xmax,y1=ymin;
    line();
    x0=a1,y0=b1,x1=c1,y1=d1,Color1=1,Color2=0,Color3=1;
    line();
    glFlush();
    cohen_sutherland();
    if(flag) {x0=a,y0=b,x1=c,y1=d,Color1=0,Color2=0,Color3=1; line();}
    glFlush();
}
int main(int argc, char *argv[])
{
    printf("Please enter two points: ");
    scanf("%d %d %d %d",&a,&b,&c,&d);
    a1=a,b1=b,c1=c,d1=d;
    glutInit(&argc, argv);
    glutInitWindowSize(800,600);
    glutInitWindowPosition(10,10);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutCreateWindow("Drawing Line at any coordant...");
    Init(800, 600);
    glutDisplayFunc(display);
    glutMainLoop();

    return 0;
}
