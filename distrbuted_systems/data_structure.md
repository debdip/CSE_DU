What is a Data Structure?
A data structure is a way of organizing the data so that the data can be used efficiently. Different kinds of data structures are 
suited to different kinds of applications, and some are highly specialized to specific tasks.For example, B-trees are particularly
well-suited for implementation of databases, while compiler implementations usually use hash tables to look up identifiers. 

Linear: A data structure is said to be linear if its elements form a sequence or a linear list.
Examples: Array. Linked List, Stacks and Queues
Non-Linear: A data structure is said to be non-linear if traversal of nodes is nonlinear in nature. Example: Graph and Trees.

What are the various operations that can be performed on different Data Structures?

Insertion − Add a new data item in the given collection of data items.
Deletion − Delete an existing data item from the given collection of data items.
Traversal − Access each data item exactly once so that it can be processed.
Searching − Find out the location of the data item if it exists in the given collection of data items.
Sorting − Arranging the data items in some order i.e. in ascending or descending order in case of numerical data and in dictionary order in case of alphanumeric data.

How is an Array different from Linked List?

-The size of the arrays is fixed, Linked Lists are Dynamic in size.
-Inserting and deleting a new element in an array of elements is expensive, Whereas both insertion and deletion can easily be done in Linked Lists.
-Random access is not allowed in Linked Listed.
-Extra memory space for a pointer is required with each element of the Linked list.
-Arrays have better cache locality that can make a pretty big difference in performance.

A linear search scans one item at a time, without jumping to any item .

The worst case complexity is  O(n), sometimes known an O(n) search
Time taken to search elements keep increasing as the number of elements are increased.

A binary search however, cut down your search to half as soon as you find middle of a sorted list.

The middle element is looked to check if it is greater than or less than the value to be searched.
Accordingly, search is done to either half of the given list

Time complexity of linear search -O(n) , Binary search has time complexity O(log n).

Which data structures is applied when dealing with a recursive function?

Recursion, which is basically a function that calls itself based on a terminating condition, makes use of the stack. Using LIFO, a call to a recursive function saves the return address so that it knows how to return to the calling function after the call terminates.

A linear search scans one item at a time, without jumping to any item .

The worst case complexity is  O(n), sometimes known an O(n) search
Time taken to search elements keep increasing as the number of elements are increased.

A binary search however, cut down your search to half as soon as you find middle of a sorted list.

The middle element is looked to check if it is greater than or less than the value to be searched.
Accordingly, search is done to either half of the given list


		Best 		average		worst	  worst(space)
Quicksort 	Ω(n log(n)) 	Θ(n log(n)) 	O(n^2) 	    O(log(n))
Mergesort 	Ω(n log(n)) 	Θ(n log(n)) 	O(n log(n)) 	O(n)
Heapsort 	Ω(n log(n)) 	Θ(n log(n)) 	O(n log(n)) 	O(1)
Bubble Sort 	Ω(n) 	          Θ(n^2) 	O(n^2) 		O(1)
Insertion Sort 	Ω(n)              Θ(n^2) 	O(n^2) 		O(1)
Selection Sort 	Ω(n^2)            Θ(n^2) 	O(n^2) 		O(1)
