import java.io.IOException;
import java.net.*;


public class ConnectionManager extends Thread {
	
	private ServerSocket serverSocket;
	private Socket socket;
	private int clientID=1;
	
	
	ConnectionManager(ServerSocket serverSocket){
		super("Conenction Manager");
		this.serverSocket=serverSocket;
	}
	
	public void run(){
		
		while(true){
			try {
				socket=serverSocket.accept();
				Server.initConnection=true;			
				ClientHandler ch=new ClientHandler("Client "+clientID++,socket);
				ch.start();
				
			} catch (IOException e) {
				
				if(e instanceof SocketException){
					System.out.println("ServerSocket is closed by main program.");
				}
				break;
				//e.printStackTrace();	
			}
		} //END of loop
		
	} //end of run
	
	
	public int getActiveThread(){
		return Thread.activeCount();
		
	}

}
