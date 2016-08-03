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
import java.util.Scanner;
import java.io.*;
class take_user_pas
{
	String get(String s)
	{
		String st;
		System.out.print(s+" ") ;
        Scanner str = new Scanner(System.in);
        st = str.nextLine();
        return st ;
	}
}
class send_message
{
	void send(String message)
	{
		String serverName = "localhost" ;
	      int port = 2300 ;
	      try
	      {
	         System.out.println("Connecting to " + serverName
	                             + " on port " + port);
	         Socket client = new Socket(serverName, port);
	         System.out.println("Connected to " + client.getRemoteSocketAddress());
	         OutputStream outToServer = client.getOutputStream();
	         DataOutputStream out = new DataOutputStream(outToServer);
	         String send = message;
	         System.out.println("Client Sent To server : "+send) ;
	         out.writeUTF(send);
	         InputStream inFromServer = client.getInputStream();
	         DataInputStream in =
	                        new DataInputStream(inFromServer);
                 if(in!=null)
	         System.out.println("Server ack to client : " + in.readUTF());
	         //client.close();
	      }catch(IOException e)
	      {
                  System.out.println("No response from server");
	          
	      }
	}
}
public class Echo_client
{
   public static void main(String [] args)
   {    while(true){
	  take_user_pas t = new take_user_pas() ;
	  String message = t.get("Enter your message : ") ;
	  send_message sm = new send_message() ;
          sm.send(message) ;
           
        }
   }
}
