import cv2
import numpy as np
import heapq as hq



#Lets first create obstacle map...  

#I will create two values for obstacle nodes. Cost value and frame intensity for visualization. 


obstacle_matrix = np.zeros((250, 400, 3))

#Circular obstacle

def create_circular_mask(h, w, center=None, radius=None):

    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)
    mask = dist_from_center <= radius
    return mask
h, w = obstacle_matrix.shape[0], obstacle_matrix.shape[1]
mask = create_circular_mask(h, w,center=(65, 300), radius = 45)         # 5 mm added for clearance.
masked_img = obstacle_matrix.copy()
masked_img[mask] = [0, 0, 210]


def get_equation_of_line(point1, point2):
    try:
        slope = (point2[0] - point1[0])/(point2[1]-point1[1])
        constant = (point1[0] - slope * point1[1])
        return slope, constant
    except:
        return point1[1], point1[1]
        
#Polygon obstacle       #I am adding 5 mm for obstacle purpose. 
p1 = (150, 105)
p2 = (65, 36)
p3 = (40, 115)
p4 = (70, 80)

clearance = 5


Y, X = np.ogrid[:250, :400]     #Image dimensions (Change this hard coded values later on to obstacle_matrix.shape[1], obstacle.shape[0] something.)
line1_p = Y - get_equation_of_line(p1, p2)[0]*X - get_equation_of_line(p1, p2)[1]
line2_p = Y - get_equation_of_line(p2, p3)[0]*X - get_equation_of_line(p2, p3)[1]
line3_p = Y - get_equation_of_line(p3, p4)[0]*X - get_equation_of_line(p3, p4)[1]
line4_p = Y - get_equation_of_line(p1, p4)[0]*X - get_equation_of_line(p1, p4)[1] 

mask_p =   ((line1_p < 5) & (line2_p+5>0) & (line3_p<5) ) | ((line1_p< 5) & (line2_p+5>0) & (line4_p+5>0))
masked_img[mask_p] = [0, 0, 210]

# cv2.circle(masked_img,(p1[1], p1[0]),3,255,-1)       #cv.circle(image, center, radius, color[, thickness)
# cv2.circle(masked_img,(p2[1], p2[0]),3,255,-1)
# cv2.circle(masked_img,(p3[1], p3[0]),3,255,-1)
# cv2.circle(masked_img,(p4[1], p4[0]),3,255,-1)

#Hexagon obstalcle.
h1 = (191, 200)
h2 = (171, 165)
h3 = (129, 165)
h4 = (109, 200)
h5 = (129, 235)
h6 = (171, 235)
line1_h = Y - get_equation_of_line(h1, h2)[0]*X - get_equation_of_line(h1, h2)[1]
line2_h = X - get_equation_of_line(h2, h3)[1]
line3_h = Y - get_equation_of_line(h3, h4)[0]*X - get_equation_of_line(h3, h4)[1]
line4_h = Y - get_equation_of_line(h4, h5)[0]*X - get_equation_of_line(h4, h5)[1]
line5_h = X - get_equation_of_line(h5, h6)[1]
line6_h = Y - get_equation_of_line(h1, h6)[0]*X - get_equation_of_line(h1, h6)[1]

mask_h =   (line1_h< 5) & (line2_h+5>0) & (line3_h+5>0) & (line4_h+5> 0) & (line5_h<5) & (line6_h<5)# &  (line1_p<0) #  &
masked_img[mask_h] = [0, 0, 210]

#Lets just make circle on the vertices of hexagon
# cv2.circle(masked_img,(h1[1], h1[0]),3,255,-1)       #cv.circle(image, center, radius, color[, thickness)
# cv2.circle(masked_img,(h2[1], h2[0]),3,255,-1)
# cv2.circle(masked_img,(h3[1], h3[0]),3,255,-1)
# cv2.circle(masked_img,(h4[1], h4[0]),3,255,-1)
# cv2.circle(masked_img,(h5[1], h5[0]),3,255,-1)
# cv2.circle(masked_img,(h6[1], h6[0]),3,255,-1)


#Masking the boarders
mask_b =   (X < 5) | (Y>245) | (X>395) | (Y< 5)
masked_img[mask_b] = [0, 0, 210]

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow('frame', masked_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()

# masked_img is the visualization of obstacle space. 
# cost_matrix is the matrix for obstacle and cost to come. and is initialised as below. 





#Create a matrix to store C2C for nodes, set -1 for obstacle nodes and ∞  for free space nodes

cost_matrix = obstacle_matrix[:,:,1].copy()

cost_matrix[cost_matrix ==0 ] = 1000000     #This should serve as infinite value. 
cost_matrix[masked_img[:, :, 2] == 210] = -1

#Based on the cost matrix itself, we can check the validity of the input points: if value is -1 then return error. 




# A code for getting a two points. 

#Get Initial (Xi) and Goal(Xg) Node from the user. 
initial_state = list(map(int, input('Please provide initial state considering origin at bottom-left corner[X, Y]... numbers seperated by space:' ).split()))
if (initial_state[0] >400) | (initial_state[1] >250):
    print('The values provided are out of bound of the region of interest, please rerun the code and provide the correct initial values again.')
if cost_matrix[250 - initial_state[1], initial_state[0]] == -1:
    print('The Initial state provided is in either the obstacle region or in boundary clearance. It is not feasible. Please rerun the code and provide the correct starting point.')

print('\nCo-ordinates of Initial Position provided is: ',initial_state)
#Modifying initial state to suit numpy and opencv convention. 
initial_state = [250 - initial_state[1], initial_state[0]]
cost_matrix[initial_state[0], initial_state[1]]  = 0



goal_state = list(map(int, input('Please provide goal state considering origin at bottom-left corner[X, Y]... numbers seperated by space:' ).split()))
if (goal_state[0] >400) | (goal_state[1] >250):
    print('The values provided are out of bound of the region of interest, please rerun the code and provide the correct goal values again.')
if cost_matrix[250 - goal_state[1], goal_state[0]] == -1:
    print('The goal state provided is in either the obstacle region or in boundary clearance. It is not feasible. Please rerun the code and provide the correct goal.')

print('\nCo-ordinates of Goal provided is: ',goal_state)
#Modifying goal state to suit numpy and opencv convention.
goal_state = [250 - goal_state[1], goal_state[0]]








#create childnode function
#[self_node_number,self_indices(y, x),  parent_node_number, [parent_node_indices(y,x)]]

action_sets = [(1,0, 1),(-1, 0, 1), (0,1, 1), (0, -1, 1), (1,1, 1.4), (-1,1, 1.4), (1,-1, 1.4), (-1,-1, 1.4)]

#I am using the len(open_list) + len(closed_list) for new node numbering. 

def create_childnode(the_node, open_list, closed_list):
    for action in action_sets:
        #we are generating a node, and 
        #Regarding this condition, I could use cost_matrix. i.e. if cost_matrix[the new node] != 1000000 then it must be a new node. 
        #because closed nodes and nodes in open_list have their cost reduced.
        #potentially_new_node_coords = [the_node[2][0]+action[0], the_node[2][1]+action[1]]

        #How do I stop algorithm from going out of dimensions?

        if ([the_node[2][0]+action[0], the_node[2][1]+action[1]] not in [closed_list[tempX_1][2] for tempX_1 in range(len(closed_list))]) & (cost_matrix[the_node[2][0]+action[0], the_node[2][1]+action[1]] != -1) & ((the_node[2][0]+action[0])< (cost_matrix.shape[0]-1)) & ((the_node[2][1]+action[1]) <(cost_matrix.shape[1]-1)) & ((the_node[2][0]+action[0])>=0) & ((the_node[2][1]+action[1]) >=0):
            if [the_node[2][0]+action[0], the_node[2][1]+action[1]] not in [open_list[tempX_2][2] for tempX_2 in range(len(open_list))]:
                
                temp_new_node = [cost_matrix[the_node[2][0], the_node[2][1]] + action[2], len(open_list)+len(closed_list)+1, [the_node[2][0]+action[0], the_node[2][1]+action[1]], the_node[1], the_node[2]]
                hq.heappush(open_list, temp_new_node)

                # open_list.append([len(open_list)+len(closed_list)+1, [the_node[1][0]+action[0], the_node[1][1]+action[1]], the_node[0], the_node[1]])

                cost_matrix[the_node[2][0]+action[0], the_node[2][1]+action[1]] = cost_matrix[the_node[2][0], the_node[2][1]] + action[2]
                masked_img[the_node[2][0]+action[0], the_node[2][1]+action[1]] = [120, 0, 0]

            #So, now only one possibility remains, this new node is in the open_list, lets check and if needed, we shall update it.
            elif cost_matrix[the_node[2][0]+action[0], the_node[2][1]+action[1]] > cost_matrix[the_node[2][0], the_node[2][1]] + action[2]: #checking if existing price is higher than the one that we are getting by this path. than updating.
                cost_matrix[the_node[2][0]+action[0], the_node[2][1]+action[1]] = cost_matrix[the_node[2][0], the_node[2][1]] + action[2]
                for i in range(len(open_list)):
                    if open_list[i][2] == [the_node[2][0]+action[0], the_node[2][1]+action[1]]:
                        open_list[i][3], open_list[i][4], open_list[i][1] = the_node[1], the_node[2], cost_matrix[the_node[2][0], the_node[2][1]] + action[2]
                # Updating parent and cost... 
                #open_list[[open_list[tempX_2][1] for tempX_2 in range(len(open_list))].index([the_node[2][0]+action[0], the_node[2][1]+action[1]])][2:4] = the_node[0], the_node[1]
    
    return True     















#creating exploration function:


#The_node is supposed to be a point under consideration. i.e. it is going to child create nodes. 
#Structure of node is: [self_node_number,self_indices(y, x),  parent_node_number, [parent_node_indices(y,x)], cost_to_come]


def queue_processing_function(open_list, closed_list):

  #  if queue_for_procesing:
    while open_list:
        #Implementing priority queue using cost_matrix.
        # temp_cost_list_0 = [open_list[temp_node_coords][1] for temp_node_coords in range(len(open_list))]
        # temp_cost_list = [cost_matrix[temp_YX[0], temp_YX[1]] for temp_YX in temp_cost_list_0]
        highest_priority_node = hq.heappop(open_list)      
        #Here, I have to use the priority queue, and I might have to change the data type of open_list. 
        #(or I can simply carry on with this seemingly inefficient algorithm.)
        closed_list.append(highest_priority_node)
        
        print(highest_priority_node)

        cv2.imshow('Exploration Visualization', masked_img)
        cv2.waitKey(1) 
        if highest_priority_node[2] == goal_state:
            cv2.waitKey(0) & 0xFF == ord('q') 
            cv2.destroyAllWindows()
    
        if highest_priority_node[2] == goal_state:      #Tis is not right. Do we need lowest cost to come to
            return highest_priority_node                        #This is the output of this function. which is to be put as an argument into generate_path function.
        else:
            create_childnode(highest_priority_node, open_list, closed_list)
            #return queue_processing_function(open_list, closed_list)     
            #How did my code worked the last time without this line? 
            #I guess, it will not get out of loop until open_list is exhausted.
        

#Backtrack function

def generate_path(node_reached_goal, path_to_goal):
    masked_img[node_reached_goal[2][0], node_reached_goal[2][1]] = [0, 200, 0]
    cv2.imshow('Exploration Visualization', masked_img)
    cv2.waitKey(50) 
    if node_reached_goal[2] == goal_state:
        cv2.waitKey(0) & 0xFF == ord('q') 
        cv2.destroyAllWindows()
    if node_reached_goal[1] == 1:
         path_to_goal.append(1)
         return None
     #Using recursion:
    else:                       
        path_to_goal.append(node_reached_goal[3])             #Appending parent index to the path_to_goal
        tempP2G = [closed_list[tempPG][1] for tempPG in range(len(closed_list))].index(node_reached_goal[3])        
        #Getting index of parent node from the closed_list.
        return generate_path(closed_list[tempP2G], path_to_goal)





# The main code:

#Data structures for nodes in open_list and closed_list
#[cost_to_come, self_node_number, selfNodeCoords[y, x], parent node, parentNodeCoords[y, x]] 
# I might not keep cost_to_come explicitly in the lists. 
 

#Initializing open_list and closed list.

open_list = []
initial_state_node = [0, 1, initial_state, 0, [0,0]]        #Taking initial_node as 1st node and its parent node as 0 and their co-ordinates as [0,0]. This parent node does not have significance. 

hq.heappush(open_list, initial_state_node)

closed_list = []

cv2.namedWindow("Exploration Visualization", cv2.WINDOW_NORMAL)

Answer_node = queue_processing_function(open_list, closed_list)      
cv2.destroyAllWindows()
print(Answer_node)

path_to_goal = []
generate_path(Answer_node, path_to_goal)
cv2.destroyAllWindows()

del path_to_goal[-1]        #1 is added twice. So removing that mannally. Because just in case if you provide goal as input, it can detect it. 
path_to_goal.sort()             #algorithm is written in a way that provides path_to_goal in reverse order.. This rectifies that. 
path_to_goal.append(Answer_node[1])      #Adding the final node's number 
print('\n\n The path to goal(This is the self node numbers): ', path_to_goal)
cv2.destroyAllWindows()


