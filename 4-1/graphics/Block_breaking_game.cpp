#include <windows.h>  // for MS Windows
#include <GL/glut.h>
#include"stdafx.h"
#include<stdlib.h>
//#include <irrKlang.h>
#include <bits/stdc++.h>
using namespace std;
int score=0,level=1,over=0,win=0,initial=0,f=1,life=10,game_over=0,total_score=0;
int pre_x=30,pre_y=30,s=0;
double speed=.5;

//string s1="score"
void drawText(const char *text,int length,int x,int y){
    glMatrixMode(GL_PROJECTION);
    double *matrix= new double[32];
    glGetDoublev(GL_PROJECTION_MATRIX,matrix);
    glLoadIdentity();
    glOrtho(0,800,0,600,-5,5);
    glMatrixMode(GL_MODELVIEW);
    glColor3f(1.0,0.0,0.0);

    glRasterPos2i(x,y);
    for(int i=0;i<length;i++){
        if(!s)
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24,(int)text[i]);
        else
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24,(int)text[i]);
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(matrix);

    }
}
void initialize()
{
	for(int n=0,x=4,y=330;n<45;n++,x+=66)
	{
	   if(x>560)
	   {
	     x=4;
	     y-=25;
	   }
	   b[n].x=x;
	   b[n].y=y;
	   b[n].w=60;
	   b[n].h=20;
	   b[n].alive=true;
	}
	for(int n=0;n<=40;n++)
        b[n].alive=false;

	block.myx=300;
	block.myy=40;
	block.width=80;
	block.height=20;
	block.lft=false;
	block.rgt=false;

	ball.ballx=20;
	ball.bally=30;
	ball.ballwh=15;
	ball.velx=0.35;
	ball.vely=0.35;

	red1=0.96;
 	green1=0.8;
 	blue1=0.69;
 	red2=0.6;
 	green2=0.8;
	blue2=0.196078;

	ball.red=0.65;
	ball.green=0.49;
	ball.blue=0.24;

	block.red=0.137255;
	block.green=0.556863;
	block.blue=0.49608;

}
bool check_collision(float Ax, float Ay, float Aw, float Ah, float Bx, float By, float Bw, float Bh) //Function for checking collision
{
  if ( Ay+Ah < By ) return false; //if A is more to the lft than B
  else if ( Ay > By+Bh ) return false; //if A is more to rgt than B
  else if ( Ax+Aw < Bx ) return false; //if A is higher than B
  else if ( Ax > Bx+Bw ) return false; //if A is lower than B

  return true; //There is a collision because none of above returned false
}
void draw1();
void draw2(){
    glColor3f(1.0,0.0,0.0);
        string s2="WELCOME TO";
        const char *c1=s2.c_str();
        drawText(c1,strlen(c1),280,303);
        string s3="BLOCK BREAKING GAME";

       const char *c3=s3.c_str();
        //cout<<c3<<endl;
        drawText(c3,strlen(c3),210,273);
        glColor3f(1.0,0.0,0.0);
          string s4="PRESS SPACE TO START";

       const char *c2=s4.c_str();
        //cout<<c3<<endl;
        drawText(c2,strlen(c2),210,233);


}
void brick1(){
    for(int i=0;i<45;i++)
        b[i].alive=false;
    int i=0;
    for( ;i<9;i++){
        b[i].alive=true;
    }
    for(i=11;i<18;i++)
        b[i].alive=true;
    for(i=21;i<27;i++)
        b[i].alive=true;
    for(i=31;i<36;i++)
        b[i].alive=true;
    for(i=41;i<45;i++)
        b[i].alive=true;
}
void brick2(){
     for(int i=0;i<45;i++)
        b[i].alive=false;
    int i=0;
    for( ;i<8;i++){
        b[i].alive=true;
    }
    for(i=9;i<16;i++)
        b[i].alive=true;
    for(i=18;i<24;i++)
        b[i].alive=true;
    for(i=27;i<32;i++)
        b[i].alive=true;
    for(i=36;i<40;i++)
        b[i].alive=true;
}
void brick3(){
    for(int i=0;i<45;i++){
        if(i%2==0)
            b[i].alive=true;
        else b[i].alive=false;
    }

}
void brick4(){
    for(int i=0;i<45;i++){
        if(!(i%2==0))
            b[i].alive=true;
        else b[i].alive=false;
    }
}
void brick5(){
    for(int i=0;i<45;i++)
        b[i].alive=false;
    b[44].alive=true;
}
void brick6(){
    for(int i=0;i<45;i++){
        if((i%4)==0)
            b[i].alive=false;
        else b[i].alive=true;
    }

}
void brick7(){
    for(int i=0;i<45;i++){
        if((i%5)==0)
            b[i].alive=false;
        else b[i].alive=true;
    }
}
void reshape()		//Modify the co-ordinates according to the key-presses and collisions etc...
{

	if(block.myx<0)
	  block.myx=0;
	if(block.myx+block.width>600)
	  block.myx=520;
	if(block.lft==true)
	  block.myx=block.myx-0.5;
	if(block.rgt==true)
	  block.myx=block.myx+0.75;



	ball.ballx+=speed*ball.velx;
	ball.bally+=speed*ball.vely;
    pre_x=ball.ballx;
    pre_y=ball.bally;
	for(int n=0;n<45;n++)
	{
	   if(b[n].alive==true)
	   {
	   	if(check_collision(ball.ballx,ball.bally,ball.ballwh,ball.ballwh,b[n].x,b[n].y,b[n].w,b[n].h)==true)
	   	{
	   	  ball.vely=-ball.vely;
	   	  b[n].alive=false;
	   	  score+=5;
	   	  total_score+=5;
	   	 // cout<<total_score<<endl;
	   	  ball.down=true;
	   	  ball.up=false;
	   	  break;
	   	}
	   }
	}
	if(ball.ballx<0)
	{
		ball.velx=-ball.velx;
		ball.right=true;
		ball.left=false;
	}
	if(ball.ballx+ball.ballwh>600)
	{
		ball.right=false;
		ball.left=true;
		ball.velx=-ball.velx;
	}
	if(ball.bally+ball.ballwh>370)
		ball.vely=-ball.vely;
	else if(ball.bally<30)
    {   over=1;
        //Sleep(7000);
        //draw1();
    }
	if(check_collision(ball.ballx,ball.bally,ball.ballwh,ball.ballwh,block.myx,block.myy,block.width,block.height)==true)
	{
			ball.vely=-ball.vely;
			ball.up=true;
			ball.down=false;
	}
	draw();
}

void specialUp(int key,int x,int y)
{
	switch(key)
	{
		case (GLUT_KEY_LEFT): block.lft=false;break;
		case (GLUT_KEY_RIGHT): block.rgt=false;break;
	}
}
void specialDown(int key,int x,int y)
{
	switch(key)
	{
		case (GLUT_KEY_LEFT): block.lft=true;break;
		case (GLUT_KEY_RIGHT): block.rgt=true;break;
	}
}
void keyboard(unsigned char key,int x,int y)
{
	if(key==27) 		//27 corresponds to the esc key
	{
		ball.velx=0;
		ball.vely=0;
		//PlaySound("test.mp3", NULL, SND_ASYNC|SND_FILENAME|SND_LOOP);	//To stop the ball from moving
		callMenu();

	}
	else if(key==48) {
    //ball.ballx=x;
	//ball.bally=y;

     ball.ballwh=12;
     ball.velx=0.35;
     ball.vely=0.35;
        callMenu();
	}
	else if (key==32){
        initial=1;
        f=1;
        if(level==1){
            win=0;
            f=1;
            brick4 ();
           block.myy=30;
        block.width=80;
        block.height=20;
        block.lft=false;
        block.rgt=false;

        ball.ballx=200;
        ball.bally=40;
        ball.ballwh=12;
        int r=rand()%3;

        ball.velx=0.30;
        ball.vely=0.5;

        ball.up=true;
        ball.down=false;
        }
        else if(level==2){
           // cout<<"keboard level 2"<<endl;
            win=0;
            f=1;
        block.myy=30;
        block.width=80;
        block.height=20;
        block.lft=false;
        block.rgt=false;

        ball.ballx=200;
        ball.bally=40;
        ball.ballwh=12;
        ball.velx=0.5;
        ball.vely=0.75;
        ball.up=true;
        ball.down=false;

             brick1();
        }
        else if (level==3){
             win=0;
            f=1;
        block.myy=30;
        block.width=80;
        block.height=20;
        block.lft=false;
        block.rgt=false;

        ball.ballx=200;
        ball.bally=40;
        ball.ballwh=12;
        ball.velx=0.45;
        ball.vely=0.5;
        ball.up=true;
        ball.down=false;

             brick2();
        }
        else if (level==4){

             win=0;
            f=1;
        block.myy=30;
        block.width=80;
        block.height=20;
        block.lft=false;
        block.rgt=false;

        ball.ballx=200;
        ball.bally=40;
        ball.ballwh=12;
        ball.velx=0.5;
        ball.vely=0.5;
        ball.up=true;
        ball.down=false;

             brick3();
        }
        else if (level==5){

             win=0;
            f=1;
        block.myy=30;
        block.width=80;
        block.height=20;
        block.lft=false;
        block.rgt=false;

        ball.ballx=200;
        ball.bally=40;
        ball.ballwh=12;
        ball.velx=0.5;
        ball.vely=0.5;
        ball.up=true;
        ball.down=false;

             brick6();
        }
        else if (level==6){

             win=0;
            f=1;
        block.myy=30;
        block.width=80;
        block.height=20;
        block.lft=false;
        block.rgt=false;

        ball.ballx=200;
        ball.bally=40;
        ball.ballwh=12;
        ball.velx=0.5;
        ball.vely=0.5;
        ball.up=true;
        ball.down=false;

             brick7();
        }
	}
}
void myinit()
{
	glViewport(0,0,600,400);
	glLoadIdentity();
	glShadeModel(GL_SMOOTH);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0,600,0,400);
}
void draw4(){
    f=1;
   // cout<<"in draw1"<<endl;
    glColor3f(1.0,0.0,0.0);
    s=1;
        string s1="Congratulations.!!";

        char temp1[100];
       const char *c1=s1.c_str();
        //cout<<c3<<endl;
        drawText(c1,strlen(c1),200,273);
        string s2="Total Score:";
        sprintf(temp1,"%d",total_score);
        s2+=temp1;
        const char *c2=s2.c_str();
        drawText(c2,strlen(c2),200,253);
        keyboard(27,0,0);

}
void draw1(){
    f=0;
   // cout<<"in draw1"<<endl;
    glColor3f(1.0,0.0,0.0);
        string s1="LEVEL ";

        //total_score+=score;
         char temp1[100],temp2[100];
         sprintf(temp1,"%d",level-1);
         s1+=temp1;
         s1+= " COMPLETE!!";

       const char *c1=s1.c_str();
        //cout<<c3<<endl;

        drawText(c1,strlen(c1),170,273);


        string s2="PRESS SPACE TO START NEXT LEVEL";
        const char *c2=s2.c_str();
                drawText(c2,strlen(c2),80,183);
        //Sleep(1000);
        score=0;
}

void draw3(){
    //cout<<"in game over"<<endl;
    int check=0;
	for(int i=0;i<45;i++)
	{
		if(b[i].alive==true)
		{
		    if(i%2==0) glColor3f(1.0,1.0,0.0);

		   else glColor3f(0.0,1.0,0.0);
		   glBegin(GL_POLYGON);
			   glVertex2f(b[i].x,b[i].y);
			   glVertex2f(b[i].x+b[i].w,b[i].y);
			   glVertex2f(b[i].x+b[i].w,b[i].y+b[i].h);
			   glVertex2f(b[i].x,b[i].y+b[i].h);
		   glEnd();
		}


 	}
	glColor3f(block.red,block.green,block.blue);
	glBegin(GL_POLYGON);
		glVertex2f(block.myx,block.myy);
		glVertex2f(block.myx+block.width,block.myy);
		glVertex2f(block.myx+block.width,block.myy+block.height);
		glVertex2f(block.myx,block.myy+block.height);
	glEnd();

	glColor3f(ball.red,ball.green,ball.blue);
	glBegin(GL_POLYGON);
		glVertex2f(ball.ballx,ball.bally);
		glVertex2f(ball.ballx+ball.ballwh,ball.bally);
		glVertex2f(ball.ballx+ball.ballwh,ball.bally+ball.ballwh);
		glVertex2f(ball.ballx,ball.bally+ball.ballwh);
	glEnd();
    glColor3f(1.0,0.0,0.0);
        string s3="GAME OVER!!";

       const char *c3=s3.c_str();

        drawText(c3,strlen(c3),300,273);

        keyboard(27,0,0);

}
void draw()		//Render the objects on the screen using the latest updated co-ordinates
{
    int check=0;
	for(int i=0;i<45;i++)
	{
		if(b[i].alive==true)
		{
		   if(i%2==0) glColor3f(1.0,1.0,0.0);

		   else glColor3f(0.0,1.0,0.0);
		   glBegin(GL_POLYGON);
			   glVertex2f(b[i].x,b[i].y);
			   glVertex2f(b[i].x+b[i].w,b[i].y);
			   glVertex2f(b[i].x+b[i].w,b[i].y+b[i].h);
			   glVertex2f(b[i].x,b[i].y+b[i].h);
		   glEnd();
		}
		else
		check++;
	}
//	int i=10;
    if(check==45&&f==1)
    {   //cout<<"what"<<endl;
        win=1;
       // if(f==1)

        level++;
        speed+=.09;
        f=0;
       // cout<<"in level"<<level<<endl;
        //cout<<"level--"<<level<<endl;
    }
    char temp[100],temp1[100],temp2[100];
   sprintf(temp, "%d", score);
   string s1="score:";
   string s2="level-";
   string s3="life:";
   s1+=temp;
    //glColor3f(block.red,block.green,block.blue);
   const char * c=s1.c_str();
    drawText(c,strlen(c),550,573);
    sprintf(temp1,"%d",level);
    s2+=temp1;
    //cout<<s2;
    const char *c1=s2.c_str();
    drawText(c1,strlen(c1),300,573);
    sprintf(temp2, "%d", life);
    s3+=temp2;
    const char *c2=s3.c_str();
    drawText(c2,strlen(c2),120,573);
//    delete[] writable;



	glColor3f(block.red,block.green,block.blue);
	glBegin(GL_POLYGON);
		glVertex2f(block.myx,block.myy);
		glVertex2f(block.myx+block.width,block.myy);
		glVertex2f(block.myx+block.width,block.myy+block.height);
		glVertex2f(block.myx,block.myy+block.height);
	glEnd();

	glColor3f(ball.red,ball.green,ball.blue);
	glBegin(GL_POLYGON);
		glVertex2f(ball.ballx,ball.bally);
		glVertex2f(ball.ballx+ball.ballwh,ball.bally);
		glVertex2f(ball.ballx+ball.ballwh,ball.bally+ball.ballwh);
		glVertex2f(ball.ballx,ball.bally+ball.ballwh);
	glEnd();
       if(over==1){
        over=0;
        if(life<=0&&f==1){

                game_over=1;
        }
        else {
            if(f==1)
            life--;
            block.myx=300;
        block.myy=40;
        block.width=80;
        block.height=20;
        block.lft=false;
        block.rgt=false;

        ball.ballx=200;
        ball.bally=32;
        ball.ballwh=12;
        ball.velx=0.5;
        ball.vely=0.5;
        ball.up=true;
        ball.down=false;

        }

        }
	glutPostRedisplay();
	glutSwapBuffers();

}
void display()
{
	glClear(GL_COLOR_BUFFER_BIT);


	glClearColor(red,green,blue,1);
	glDisable(GL_DEPTH_TEST);
	//cout<<"in the display"<<endl;
	if(!initial){
        draw2();
	}
	else if(win==1&&level<=6||f==0&&level<=6)
        draw1();
    else if(game_over==1)
        draw3();
    else if(level>6)
        draw4();
    else if(f==1){
        //cout<<"in draw"<<endl;
        draw();
    }
	glFlush();
	reshape();
}
void bg1_menu(int opt)
{
	switch(opt)
	{
		case RED:	red=1.0;
				green=0.0;
				blue=0.0;
				display();
				break;
		case GREEN:	red=0.0;
				green=1.0;
				blue=0.0;
				display();
			 	break;
		case BLUE:	red=0.0;
				green=0.0;
				blue=1.0;
				display();
			 	break;
		case BLACK:	red=0.0;
				green=0.0;
				blue=0.0;
				display();
			 	break;
	}

}
void bg2_menu(int opt)
{
	switch(opt)
	{
		case GOLD:	ball.red=0.858824;
			  	ball.green=0.858824;
			  	ball.blue=0.439216;
			  	break;

		case ORCHID:	ball.red=0.858824;
			    	ball.green=0.439216;
			    	ball.blue=0.858824;
			    	break;
	}
}
void bg3_menu(int opt)
{
	switch(opt)
	{
		case C1:
			red1=0.72;
			green1=0.45;
			blue1=0.20;
			red2=0.42;
			green2=0.26;
			blue2=0.15;
			break;
		case C2:
			red1=1;
			green1=0.5;
			blue1=0;
			red2=0.9;
			green2=0.91;
			blue2=0.98;
			break;
		case C3:
			red1=0.858824;
			green1=0.439216;
			blue1=0.858824;
			red2=0.36;
			green2=0.2;
			blue2=0.09;
	}
}
void bg4_menu(int opt)
{
	switch(opt)
	{
		case BLACK1:
			block.red=0.0;
			block.green=0.0;
			block.blue=0.0;
			break;
		case WHITE:
			block.red=1;
			block.green=1;
			block.blue=1;
			break;
	}
}
void sp_menu(int opt)
{
	switch(opt)
	{
		case INC:
			//ball.velx++;
			//ball.vely++;
			break;
		case DEC:
			//ball.velx-=0.5;
			//ball.vely-=0.5;
			break;

	}
}
void main_menu(int opt)
{
	switch(opt)
	{
		case CONTINUE: revert();
			       break;
		case QUIT: {
            exit(0);
		}
	}
}
void callMenu()
{
	int bg,bg1,bg2,bg3,bg4,sp;
	bg1=glutCreateMenu(bg1_menu);
	glutAddMenuEntry("Red",RED);
	glutAddMenuEntry("Green",GREEN);
	glutAddMenuEntry("Blue",BLUE);
	glutAddMenuEntry("Default",BLACK);

	bg2=glutCreateMenu(bg2_menu);
	glutAddMenuEntry("gold",GOLD);
	glutAddMenuEntry("Orchid",ORCHID);

	bg3=glutCreateMenu(bg3_menu);
	glutAddMenuEntry("Combo1",C1);
	glutAddMenuEntry("Combo2",C2);
	glutAddMenuEntry("Combo3",C3);

	bg4=glutCreateMenu(bg4_menu);
	glutAddMenuEntry("Black",BLACK1);
	glutAddMenuEntry("White",WHITE);

	sp=glutCreateMenu(sp_menu);
	glutAddMenuEntry("Increase",INC);
	glutAddMenuEntry("Decrease",DEC);

	bg=glutCreateMenu(main_menu);
	glutAddSubMenu("Background",bg1);
	glutAddSubMenu("Ball",bg2);
	glutAddSubMenu("Brick",bg3);
	glutAddSubMenu("Block",bg4);

	glutCreateMenu(main_menu);
	glutAddMenuEntry("Continue",CONTINUE);
	glutAddSubMenu("Color",bg);
	glutAddSubMenu("Speed",sp);
	glutAddMenuEntry("Quit Game",QUIT);

	glutAttachMenu(GLUT_RIGHT_BUTTON);


}
void processmenu(int opt)
{
	switch(opt)
	{
		case CONTINUE:
			revert();
			break;
		case INC:
			ball.velx+=2;
			ball.vely+=2;
			break;

		case QUIT: exit(0);
	}

}
void revert()
{
	ball.velx=1.05;
	ball.vely=1.5;
	if(ball.up==true)
	{
		if(ball.right==true)
		{
			ball.velx=ball.velx;
			ball.vely=ball.vely;
		}
		else if(ball.left==true)
		{
			ball.velx=-ball.velx;
			ball.vely=ball.vely;
		}
	}
	else if(ball.down=true)
	{
		if(ball.right=true)
		{
			ball.velx=ball.velx;
			ball.vely=-ball.vely;
		}
		else if(ball.left==true)
		{
			ball.velx=-ball.velx;
			ball.vely=-ball.vely;
		}
	}
}
int main(int argc,char *argv[])
{
	glutInit(&argc,argv);
	glutInitWindowSize(500,600);
	glutInitWindowPosition(100,100);
	glutCreateWindow("Block Breaker");
	initialize();
	myinit();
	draw();
	glutDisplayFunc(display);
        glutSpecialFunc(specialDown);
    	glutSpecialUpFunc(specialUp);
    	glutKeyboardFunc(keyboard);
	glutIdleFunc(reshape);
	glutMainLoop();
	return 0;
}
