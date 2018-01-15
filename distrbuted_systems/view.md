Define Bandwidth and Latency?

 Bandwidth of a network is given by the number of bits that can be transmitted over the network in a certain period of time. 
 Latency corresponds to how long it takes a message to travel from one end off a network to the other. 
 It is strictly measured in terms of time.
 
 Routing?
 
 Routing is the process of moving packets across a network from one host to a another.
 It is usually performed by dedicated devices called routers. 
 
 RTD/RTT--round-trip delay time (RTD) or round-trip time (RTT) is the length of time it takes for a signal to be sent,
 plus the length of time it takes for an acknowledgment of that signal to be received.
 This time delay therefore consists of the propagation times between the two points of a signal.
 
 Define the terms Unicasting, Multiccasting and Broadcasting?
 If the message is sent from a source to a single destination node, it is called Unicasting.
If the message is sent to some subset of other nodes, it is called Multicasting.
If the message is sent to all the m nodes in the network it is called Broadcasting.


subnet mask?
A subnet mask is a 32- or 128-bit number that segments an existing IP address in a TCP/IP network and divides that address into discrete network and host addresses. The process of subnetting can further divide the host portion of an IP address into additional subnets to route traffic within the larger subnet.
255.255.0.0---network(255.255).host(0.0)

OSI model(7 layer model)--

The Open Systems Interconnection model (OSI model)

Physical-(bit) Transmission and reception of raw bit streams over a physical medium
Data link-(Frame)Reliable transmission of data frames between two nodes connected by a physical layer

Network-(packet) Structuring and managing multi-node network, including addressing, routing and traffic control

Transport-Segment(TCP)/Datagram(UDP) Reliable transmission of data segments between points of network.

Session-(Data)managing communication session, continuous exchange of information in the form of multiple back and forth transmission between two nodes

Presentation-(Data) Translation of data between a network service and an application; including character encoding, data compression and encryption/decrytion

Application-(Data)-High level APIs, including resource sharing


5 layer model--


Physical-(bit) Transmission and reception of raw bit streams over a physical medium --Addressing(N/A)
Data link-(Frame)Reliable transmission of data frames between two nodes connected by a physical layer---Addressing(MAC)

Network/internet-(packet) Structuring and managing multi-node network, including addressing, routing and traffic control---Addressing(IP)

Transport-Segment(TCP)/Datagram(UDP) Reliable transmission of data segments between points of network.---Addressing(port)

Application-(Data)-High level APIs, including resource sharing---(N/A)

What is Bit Stuffing?
Bit stuffing is the process of adding one extra 0 whenever five consecutive Is follow a 0 in the data, so that the receiver does not mistake the pattern 0111110 for a flag.

What is Flow Control?
Flow control refers to a set of procedures used to restrict the amount of data that the sender can send before waiting for acknowledgment.

What is Error Control ?
Error control is both error detection and error correction. It allows the receiver to inform the sender of any frames lost or damaged in transmission and coordinates the retransmission of those frames by the sender. 

Automatic repeat request (ARQ) is a protocol for error control in data transmission. When the receiver detects an error in a packet, it automatically requests the transmitter to resend the packet.

What is point-to-point protocol?
A communications protocol used to connect computers to remote networking services including Internet service providers.

What is attenuation?
The degeneration of a signal over distance on a network cable is called attenuation.

What is MAC address?
The address for a device as it is identified at the Media Access Control (MAC) layer in the network architecture. MAC address is usually stored in ROM on the network adapter card and is unique.


Difference between bit rate and baud rate?
Bit rate is the number of bits transmitted during one second whereas baud rate refers to the number of signal units per second that are required to represent those bits. 
  baud rate = (bit rate / N) 
  where N is no-of-bits represented by each signal shift.

Transmission Control Protocol (TCP)--
provides reliable, ordered, and error-checked delivery of a stream of octets between applications running on hosts communicating by an IP network.

Applications that do not require reliable data stream service may use the User Datagram Protocol

What is SMTP?

SMTP is short for Simple Mail Transfer Protocol. This protocol deals with all Internal mail, and provides the necessary mail delivery services on the TCP/IP protocol stack.

What is DHCP?

DHCP is short for Dynamic Host Configuration Protocol. Its main task is to automatically assign an IP address to devices across the network. It first checks for the next available address not yet taken by any device, then assigns this to a network device.


Briefly describe NAT.

NAT is Network Address Translation. This is a protocol that provides a way for multiple computers on a common network to share single connection to the Internet.

Time-to-live (TTL) is a value in an Internet Protocol (IP) packet that tells a network router whether or not the packet has been in the network too long and should be discarded.

network throughput is the amount of data moved successfully from one place to another in a given time period, and typically measured in bits per second (bps), as in megabits per second (Mbps) or gigabits per second (Gbps)



