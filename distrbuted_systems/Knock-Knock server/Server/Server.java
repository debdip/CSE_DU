import java.io.*;
import java.net.*;
import java.sql.ResultSet;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;


public class Server extends Thread {

	public static int numberOfClient=0; //To keep track number of active clients
	
	public static boolean initConnection=false; //This flag will change after first connection
	
	
	public static void main(String[] args) throws IOException {
		
		System.out.println("<-----Server---->");
		
		Scanner sc=new Scanner(System.in);
		
		int port;
		System.out.println("Please specify a port number larger than 1024: ");
		
		port=sc.nextInt();
		
		ServerSocket serverSocket=new ServerSocket(port);
		
		System.out.println("Server is waiting at port :"+serverSocket.getLocalPort());
		
		//DbHandler db=new DbHandler();
		//db.prinAllJokes();

		
		ConnectionManager cm=new ConnectionManager(serverSocket);
		cm.start();
		
		boolean socketClosed=false;
	    
		while(true){
			
			
			try { //small pause to initiate connection
				TimeUnit.MILLISECONDS.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			
//			System.out.println("No client joined yet!");
//			System.out.println("Connection status : "+Server.initConnection);
			
			
			if(Server.initConnection==true){
				
				try { //pause to update numberOfClients
					TimeUnit.SECONDS.sleep(1);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				
		    	  
				 while(Server.numberOfClient>0){
					 
					 try {
							TimeUnit.MILLISECONDS.sleep(100);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
					 }
		    	 }
		    	   
		    	   
		    	 System.out.println("There are no more clients to serve.");
		    	   
		    	   //No more active client
		    	 serverSocket.close();
		    	 socketClosed=true;
		    	 break;
		    }//ENDIF
			
			if(socketClosed) break;
		}
		
		System.out.println("Server is closing now");
        
	} //END OF main
	
}
