/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package tcp_chat;

import Tcp_chat.Receiver;
import Tcp_chat.Sender;
import java.net.*;


/**
 *
 * @author dipdeb
 */
public class Server {
    
    public static void main(String args[]) throws Exception{
        
            
        int port=9876;
    
        ServerSocket serverSocket=new ServerSocket(port);
            
        System.out.println("Server waiting for client on port: "+ port);
            
        Socket socket=serverSocket.accept();
            
        System.out.println("Server connected to: "+socket.getRemoteSocketAddress());
            
            
        Sender sender=new Sender(socket,"Server");
        Receiver receiver=new Receiver(socket,"Server");
            
            
            
        sender.start();
        
        receiver.start();
            
    }
    
}
