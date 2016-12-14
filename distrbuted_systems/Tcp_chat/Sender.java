/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Tcp_chat;

import java.io.*;
import java.net.*;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author dipdeb
 */
public class Sender extends Thread {
    
    Socket socket;
    String name;
    DataOutputStream outputStream;
    Scanner scanner;
    
    
    
    public Sender(Socket socket, String name){
        
        this.name=name;
        this.socket=this.socket;
        scanner=new Scanner(System.in);
        try {
            outputStream = new DataOutputStream(socket.getOutputStream());
        } catch (IOException ex) {
            Logger.getLogger(Sender.class.getName()).log(Level.SEVERE, null, ex);
        }
    
    }

    @Override
    public void run() {
        while (true) {
			try {
				String message = scanner.nextLine();
				outputStream.writeUTF(name + ": " + message);
			} catch (Exception ex) {
				System.out.println("Connection lost " );
			}
		}
    }
    
}
