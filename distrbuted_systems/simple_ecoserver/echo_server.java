/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package echo_server;

/**
 *
 * @author dip
 */
import java.net.*;
import java.io.*;

public class Echo_server extends Thread
{
   private ServerSocket serverSocket;
   
   public Echo_server(int port) throws IOException
   {
      serverSocket = new ServerSocket(port);
      
   }

   public void run()
   { int check=0,end=0;
      while(true)
      {
         try
         {
            System.out.println("Waiting for client on port " +serverSocket.getLocalPort() + "...");
            Socket server = serverSocket.accept();
            System.out.println("Connected to "+ server.getRemoteSocketAddress());
            DataInputStream in =new DataInputStream(server.getInputStream());
            
            String st=in.readUTF();
            if(st.equals("BEGIN"))  
               check=1;   // for Starting to send ack 
            System.out.println("Received : "+st) ;
            if(st.equals("END")&&check==1)  // for stopping communication after sendind End ack
            {
                check=0;
                end=1;
                DataOutputStream out =new DataOutputStream(server.getOutputStream());
            out.writeUTF("End ack" );
                
            }
           if(check==1&&end==0){
            DataOutputStream out =new DataOutputStream(server.getOutputStream());
            out.writeUTF(st);
           }
            
            server.close();
         } catch(IOException e)
         {
            e.printStackTrace();
            break;
         }
      }
      
   }
   public static void main(String [] args)
   {
      int port = 2300;
      try
      {
         Thread t = new Echo_server(port);
         t.start();
      }catch(IOException e)
      {
         e.printStackTrace();
      }
   }
}
