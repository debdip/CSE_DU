import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import java.util.Scanner;


public class Client {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		System.out.println("<-----Client Program----->");
		
		int port=9896; //Default
        
        String ipAddr="localhost"; //ip address of server
        
        Scanner scanner=new Scanner(System.in);
        
		System.out.println("Please eneter server's IP address, press 1 for localhost: ");
		
		ipAddr=scanner.nextLine();
		
		if(ipAddr.equals("1")) ipAddr="localhost";
		
		System.out.println("Please eneter server's port number: ");
		
		port=scanner.nextInt();
		
		
         
        Socket socket;
         
        try{
        	
        	socket=new Socket(ipAddr,port);
        	DataOutputStream outputStream=new DataOutputStream(socket.getOutputStream());
 			DataInputStream inputStream=new DataInputStream(socket.getInputStream());
 			Scanner sc=new Scanner(System.in);
 			
 			while(true){
 				String msg=inputStream.readUTF();
 			
 				System.out.println("Server: "+msg);
 				
 				if(msg.equalsIgnoreCase("I have no more jokes to tell.")){

 					throw new  Exception("No more jokes");
 				}
 				
 				String notice=inputStream.readUTF();
 				if(notice.equalsIgnoreCase("Would you like to listen to another? (Y/N)")){
 					System.out.println("Server: "+notice);
 				}
 				else{
 					//ignore
 				}
 				System.out.print("Client: ");
 				String reply=sc.nextLine();
 				outputStream.writeUTF(reply);
 				
 			}
        	 
        }catch(Exception e){
        		
        		System.out.println("Client is disconnected from server.");
        	
        }

	}

}
