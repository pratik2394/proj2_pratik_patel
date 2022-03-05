# proj2_pratik_patel
This is a part of the coursework of ENPM661. This is to fulfill requirement of project 2: The implementation of Dijkstra algorithm. 

In final graph, color coding is as follow.

Red: Obstacles
Blue: exploration... (Open_list)
Green: Generated Path. 


# Please note that I am using default values for 0-indexed language. i.e. our cost matrix of shape 250 * 400 that represents region can only take values upto. 249, 399. Please input initial and goal node accordingly. 


The visualisation of exploration is done along with the exploration. i.e. generation of new nodes.
Once the goal node has been reached, the code will stop generating new nodes and you need to press 'Enter' in order to shut the exploration visualization window. Then the code will generate the path. (Press 'Enter' again for code to run the path generation.)



#Please note that, the nodes that are being printed out on the terminal are priority queue poped items. And they contain co-ordinates converted into the numpy/opencv co-ordinate system. i.e. [row, column] with origin at the top-left side. so, if you input [x = 5, y =5] as a initial node, then it will start exploration from [y=245, x=5] and will display node as such. 


#I kept this just as it is for my convenience and that might help you understand the number of the node being generated. And grasp the progress of the code. 

I tested code without visualisation. and timing is as below. 
#Code takes less than 10 minutes to explore 40,000 nodes. (The environment given is 250*400, hence it is not possible to generate nodes above 100,000.) And as it reaches greated number,  it will slow down. Please give it time, if you are going to insert test cases that are far apart. 


If the code is too slow, please comment out the visualisation part from exploration code. (i.e. visualization code in generate_child function. #you can see the comment in the code. 
