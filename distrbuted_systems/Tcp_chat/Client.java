package Tcp_chat;

import java.net.*;
import java.util.*;
import Tcp_chat.Receiver;
import Tcp_chat.Sender;

/**
 *
 * @author dipdeb
 */


public class Client {
    
    public static void main(String args[]) throws Exception{
            
            int port=9876;  
            
            String ipAddr="localhost"; //ip address of server
            
            Socket socket=new Socket(ipAddr,port);
            
            System.out.println("Client connected to " + socket.getRemoteSocketAddress());
            
            
            Sender sender=new Sender(socket,"Client");
            Receiver receiver=new Receiver(socket,"Client");
            
            sender.start();
            
            receiver.start();
                    
         
    
    }
    
       
}
