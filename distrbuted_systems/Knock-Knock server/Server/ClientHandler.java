import java.net.*;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.io.*;

public class ClientHandler extends Thread {
	
	private Socket socket;
	private DataOutputStream outputStream;
	private DataInputStream inputStream;
	private DbHandler db;
	private int maxJokes=30;
	private ArrayList <Integer> toldJokes;
	boolean moreJokes=false;
	
	 
	ClientHandler(String name,Socket socket){
		super(name);
		Server.numberOfClient++;
		
		System.out.println(this.getName()+" joined.");
		System.out.println("Number of active clients : "+Server.numberOfClient);
		
		this.socket=socket;
		this.toldJokes=new ArrayList <Integer>();
		try {
			this.outputStream=new DataOutputStream(socket.getOutputStream());
			this.inputStream=new DataInputStream(socket.getInputStream());
			this.db=new DbHandler();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	
	
	@Override
    public void run() {
		
		while(toldJokes.size()<=maxJokes-1){
			
			Random rand = new Random();
			
			int id=rand.nextInt(maxJokes)+1;
			
			while(true){
				if(toldJokes.contains(id)){
					id=rand.nextInt(maxJokes)+1;
					System.out.println("id "+id);
				}else{
					toldJokes.add(id);
					break;
				}
			}//we got our uniqe joke id
			
			System.out.println("New joke id for "+this.getName()+" is :"+id);
			
			ResultSet rs=db.getJokesByID(id);
			int i=2;
			
			while(true){ //Jokes Start
				
				try{
					outputStream.writeUTF(rs.getString(i));
					i++;
					if(i>6){
						outputStream.writeUTF("Would you like to listen to another? (Y/N)");
						String re=inputStream.readUTF();
						if(re.equalsIgnoreCase("Y")) {
							moreJokes=true;
							break;
						}else{
							moreJokes=false;
							break;
							
						}
						
					}
					else{
						outputStream.writeUTF(" ");
					}
					
					String reply=inputStream.readUTF();
					
					while(true){
						if(reply.equalsIgnoreCase(rs.getString(i))==false){
							
							outputStream.writeUTF("You are supposed to say,\""+rs.getString(i)+"\". Let’s try again.");
							outputStream.writeUTF(" ");
							reply=inputStream.readUTF();
						}
						else{
							i++;
							break;
						}
					}
					
				}catch(Exception e){
					
				}
				
				
			} // This loop sends one joke at a time
			
			if(moreJokes==false) break;
			
		}//lop for all jokes;
		
		if(toldJokes.size()==maxJokes){
			try{
				outputStream.writeUTF("I have no more jokes to tell.");
				TimeUnit.SECONDS.sleep(1);
			}catch(Exception e){
				
			}
		}
		
		
		try {
			this.socket.close();
			Server.numberOfClient--;
			System.out.println(this.getName()+" left.");
			System.out.println("After leaving noc : "+Server.numberOfClient);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
    }
	

	void println(Object obj){
		System.out.println(obj);
	}
	

}
