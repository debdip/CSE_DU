/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Tcp_chat;
import java.io.*;
import java.net.*;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author dipdeb
 */
public class Receiver extends Thread {
    
    Socket socket;
    String name;
    DataInputStream in_Stream;
    
    
    public Receiver(Socket socket, String name){
        this.name=name;
        this.socket=socket;
        try {
            in_Stream = new DataInputStream(socket.getInputStream());
        } catch (IOException ex) {
            Logger.getLogger(Receiver.class.getName()).log(Level.SEVERE, null, ex);
        }
    
    
    }

    @Override
    public void run() {
        while (true) {
			try {
				String message = in_Stream.readUTF();
				System.out.println(message);
			} catch (Exception ex) {
				System.out.println("Connection lost." );
                                
			}
		}
       
    }
    
    
    
}
