import java.sql.*;

public class DbHandler {
	
	Connection c = null;
	
	public DbHandler(){
		
	    try {
	      Class.forName("org.sqlite.JDBC");
	      c = DriverManager.getConnection("jdbc:sqlite:database/jokes.db");
	    }catch(Exception e){
	    	e.printStackTrace();
	    }
	}
	
	public ResultSet getJokesByID(int id){
		
		Statement stmt;
		try {
			stmt = c.createStatement();
			String query="SELECT * FROM Jokes where \"ID\"="+id+";";
//			System.out.println(query);
			ResultSet rs = stmt.executeQuery(query);
			return rs;
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;	    
	}
	
	public void prinAllJokes(){
		
		Statement stmt;
		
		try{
			stmt = c.createStatement();
			String query="SELECT * FROM Jokes;";
			ResultSet rs = stmt.executeQuery(query);
			
			while(rs.next()){
				  System.out.println("Joke ID : "+rs.getInt(1));
		    	  System.out.println(rs.getString(2));
		    	  System.out.println(rs.getString(3));
		    	  System.out.println(rs.getString(4));
		    	  System.out.println(rs.getString(5));
		    	  System.out.println(rs.getString(6));
		    	  System.out.println("\n");
		      }
			
		}catch(Exception e){
			
		}
	}

}
