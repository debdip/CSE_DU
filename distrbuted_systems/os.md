mutex-

semaphore-

monitor-A monitor consists of a mutex (lock) object and condition variables.
A condition variable is basically a container of threads that are waiting for a certain condition.

busy waiting-

context switch-  a context switch occurs when the time slice for one process has expired and a new process is to be loaded
from the ready queue. This will be instigated by a timer interrupt,
which will then cause the current process's state to be saved and the new process's state to be restored.

Coffman conditions deadlock:

Mutual exclusion condition: a resource cannot be used by more than one process at a time
Hold and wait condition: processes already holding resources may request new resources
No preemption condition: only a process holding a resource may release it
Circular wait condition: two or more processes form a circular chain where each process waits for a resource that the next process in the chain holds

There are three basic approaches to recovery from deadlock:
Inform the system operator, and allow him/her to take manual intervention.
Terminate one or more processes involved in the deadlock
Preempt resources.

Ostrich algorithm:


Banker's algorithm:Bankers’s Algorithm is resource allocation and deadlock avoidance algorithm which test all the request made by processes for resources, it check for safe state, if after granting request system remains in the safe state it allows the request and if their is no safe state it don’t allow the request made by the process.

Inputs to Banker’s Algorithm
1. Max need of resources by each process.
2. Currently allocated resources by each process.
3. Max free available resources in the system.

Request will only be granted under below condition.
1. If request made by process is less than equal to max need to that process.

2. If request made by process is less than equal to freely availbale resource in the system.



Any solution to the critical section problem must satisfy three requirements:

Mutual Exclusion : If a process is executing in its critical section, then no other process is allowed to execute in the critical section.
Progress : If no process is in the critical section, then no other process from outside can block it from entering the critical section.
Bounded Waiting : A bound must exist on the number of times that other processes are allowed to enter their critical sections after a process has made a request to enter its critical section and before that request is granted.

Peterson’s Solution
Peterson’s Solution is a classical software based solution to the critical section problem.

In Peterson’s solution, we have two shared variables:

boolean flag[i] :Initialized to FALSE, initially no one is interested in entering the critical section
int turn : The process whose turn is to enter the critical section.




A scheduling algorithm is the algorithm which dictates how much CPU time is allocated to Processes and Threads:
CPU scheduling decisions take place under one of four conditions:
When a process switches from the running state to the waiting state, such as for an I/O request or invocation of the wait( ) system call.(new process must)
When a process switches from the running state to the ready state, for example in response to an interrupt.(depends)
When a process switches from the waiting state to the ready state, say at completion of I/O or a return from wait( ).(depends)
When a process terminates.(new process must)


algorithm:
first come, first serve:

shortest job first:pick the quickest fastest little job that needs to be done, get it out of the way first, and then pick the next smallest fastest job to do next.

priority scheduling: each job is assigned a priority and the job with the highest priority gets scheduled first.
Priority scheduling can suffer from a major problem known as indefinite blocking, or starvation, in which a low-priority task can wait forever because there are always some other jobs around that have higher priority.

Round robin:Round robin scheduling is similar to FCFS scheduling, except that CPU bursts are assigned with limits called time quantum.
When a process is given the CPU, a timer is set for whatever value has been set for a time quantum.
If the process finishes its burst before the time quantum timer expires, then it is swapped out of the CPU just like the normal FCFS algorithm.
If the timer goes off first, then the process is swapped out of the CPU and moved to the back end of the ready queue.


process state-
new, ready, running, waiting, terminated

 a context switch occurs when the time slice for one process has expired and a new process is to be loaded from the ready queue. This will be instigated by a timer interrupt, which will then cause the current process's state to be saved and the new process's state to be restored.


Thread-A thread is a path of execution within a process. Also, a process can contain multiple threads.Thread is also known as lightweight process. The idea is achieve parallelism by dividing a process into multiple threads. For example, in a browser, multiple tabs can be different threads. MS word uses multiple threads, one thread to format the text, other thread to process inputs etc. 

Advantages of Thread over Process
1. Responsiveness: If the process is divided into multiple threads, if one thread completed its execution, then its output can be immediately responded.

2. Faster context switch: Context switch time between threads is less compared to process context switch. Process context switch is more overhead for CPU.

3. Effective Utilization of Multiprocessor system: If we have multiple threads in a single process, then we can schedule multiple threads on multiple processor. This will make process execution faster.

4. Resource sharing: Resources like code, data and file can be shared among all threads within a process.

Paging

Paging is a memory management scheme that allows processes physical memory to be discontinuous, and which eliminates problems with fragmentation by allocating memory in equal sized blocks known as pages.
Paging eliminates most of the problems of the other methods discussed previously, and is the predominant memory management technique used today.

A translation lookaside buffer (TLB) is a memory cache that is used to reduce the time taken to access a user memory location. It is a part of the chip's memory-management unit (MMU). The TLB stores the recent translations of virtual memory to physical memory and can be called an address-translation cache.

DLL is a dynamic link library file format used for holding multiple codes and procedures for Windows programs. DLL files were created so that multiple programs could use their information at the same time, aiding memory conservation.




https://www.cs.uic.edu/~jbell/CourseNotes/OperatingSystems/9_VirtualMemory.html
https://en.wikipedia.org/wiki/Page_fault

